# DDPM training and sampling wrapper for trajectory generation.

import torch
import torch.nn as nn
from typing import Optional, Dict
from tqdm.auto import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.utils.config import DataDict
from .generator_cmag import CMAGNNGenerator
from src.loss import Loss

class DiffusionModel(nn.Module):
    def __init__(self, cfg, generator_network, **kwargs):
        super(DiffusionModel, self).__init__()
        self.cfg = cfg
        self.generator = generator_network
        diffusion_cfg = self.cfg.model.diffusion
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_cfg.get('timesteps', 100),
            beta_schedule='squaredcos_cap_v2',
            prediction_type="epsilon" 
        )
        self.loss_fn = Loss(self.cfg)
        self.sampling_timesteps = diffusion_cfg.get('sampling_timesteps', 100)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("DiffusionModel should be used through training_loss() or sample().")

    def training_loss(self, gt_traj_normalized: torch.Tensor, gt_traj_absolute: torch.Tensor, 
                      scale: torch.Tensor, c_env: torch.Tensor, c_target: torch.Tensor, c_start: torch.Tensor,
                      true_start_point: torch.Tensor, node_embeddings: torch.Tensor, 
                      node_pos: torch.Tensor, batch_indices: torch.Tensor,
                      epoch: int = 0) -> Dict[str, torch.Tensor]:
        
        device = gt_traj_normalized.device
        B, M, C = gt_traj_normalized.shape
        
        noise = torch.randn_like(gt_traj_normalized)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
        x_t_normalized = self.noise_scheduler.add_noise(gt_traj_normalized, noise, t)

        predicted_x0 = self.generator(
            noisy_traj=x_t_normalized, 
            t=t,
            c_env=c_env, 
            c_target=c_target, 
            c_start=c_start,
            true_start_point=true_start_point,
            node_embeddings=node_embeddings,
            node_pos=node_pos,
            batch_indices=batch_indices,
            scale=scale
        )

        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        sqrt_alpha_prod = alphas_cumprod[t] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(predicted_x0.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(predicted_x0.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        pred_noise = (x_t_normalized - sqrt_alpha_prod * predicted_x0) / sqrt_one_minus_alpha_prod
        
        simple_loss = nn.functional.mse_loss(pred_noise, noise)
        loss_info = {'L_simple': simple_loss}

        aux_loss_info = self.loss_fn(
            predicted_x0=predicted_x0,
            gt_traj_normalized=gt_traj_normalized,
            gt_traj_absolute=gt_traj_absolute,
            point_cloud=node_pos,
            batch_indices=batch_indices,
            epoch=epoch
        )
        
        total_loss = simple_loss + aux_loss_info.pop('loss')
        loss_info.update(aux_loss_info)
        loss_info['loss'] = total_loss
        
        final_output = loss_info
        final_output['pred'] = predicted_x0.detach()
        final_output[DataDict.path] = gt_traj_absolute
        return final_output
    
    @torch.no_grad()
    def sample(self, c_env: torch.Tensor, c_target: torch.Tensor, c_start: torch.Tensor, 
               true_start_point: torch.Tensor, node_embeddings: torch.Tensor, 
               node_pos: torch.Tensor, batch_indices: torch.Tensor, scale: torch.Tensor,
               target_conditioning: str = "soft", hard_target_point: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        device = next(self.generator.parameters()).device
        B = c_env.shape[0]
        M = self.cfg.model.diffusion.waypoints_num
        shape = (B, M, 2)
        
        # Sampling starts from isotropic Gaussian noise.
        x_t = torch.randn(shape, device=device)

        if target_conditioning not in {"soft", "hard"}:
            raise ValueError(f"Unsupported target_conditioning: {target_conditioning}")
        use_hard_conditioning = target_conditioning == "hard"
        if use_hard_conditioning:
            if hard_target_point is None:
                raise ValueError("hard_target_point must be provided when target_conditioning='hard'")
            x_t[:, -1, :] = hard_target_point

        self.noise_scheduler.set_timesteps(self.sampling_timesteps)
        
        for t in tqdm(self.noise_scheduler.timesteps, desc='Sampling', total=self.sampling_timesteps, leave=False):
            t_batch = t.repeat(B).to(device)
            if use_hard_conditioning:
                x_t[:, -1, :] = hard_target_point
            
            pred_x0 = self.generator(
                noisy_traj=x_t, 
                t=t_batch,
                c_env=c_env, 
                c_target=c_target, 
                c_start=c_start,
                true_start_point=true_start_point,
                node_embeddings=node_embeddings,
                node_pos=node_pos,
                batch_indices=batch_indices,
                scale=scale
            )
            if use_hard_conditioning:
                pred_x0[:, -1, :] = hard_target_point
            
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
            sqrt_alpha_prod = alphas_cumprod[t_batch] ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            while len(sqrt_alpha_prod.shape) < len(pred_x0.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    
            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t_batch]) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(pred_x0.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    
            pred_epsilon = (x_t - sqrt_alpha_prod * pred_x0) / sqrt_one_minus_alpha_prod

            x_t = self.noise_scheduler.step(pred_epsilon, t, x_t).prev_sample
            if use_hard_conditioning:
                x_t[:, -1, :] = hard_target_point
        if use_hard_conditioning:
            x_t[:, -1, :] = hard_target_point
        return x_t
