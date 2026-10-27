# Top-level model orchestration for training and trajectory sampling.

import torch
import torch.nn as nn
from typing import Dict, Any

from src.models.perception import PerceptionEncoder, TargetEncoder, StartPointEncoder
from src.models.diffusion import DiffusionModel
from src.models.generator_cmag import CMAGNNGenerator
from src.utils.config import DataDict, GeneratorType

class HNav(nn.Module):
    def __init__(self, config, device):
        super(HNav, self).__init__()
        self.config = config
        self.device = device
        model_cfg = config.model
        dataset_cfg = config.dataset

        self.perception = PerceptionEncoder(model_cfg.perception)
        self.target_encoder = TargetEncoder(
            input_dim=2,
            hidden_dim=model_cfg.perception.target_cfg.get('hidden_dim', 64),
            out_dim=model_cfg.perception.target_cfg.get('out_dim', 32)
        )
        self.start_point_encoder = StartPointEncoder(
            input_dim=2,
            hidden_dim=64,
            out_dim=32
        )
        self.generator_type = getattr(model_cfg, "generator_type", GeneratorType.diffusion)
        if self.generator_type == GeneratorType.diffusion:
            self.generator = CMAGNNGenerator(model_cfg)
            self.diffusion_model = DiffusionModel(self.config, self.generator)
        else:
            raise NotImplementedError("Only Diffusion model is fully supported in this version.")

    def _get_batch_field(self, batch: Dict[str, Any], primary: str, fallbacks=None):
        if fallbacks is None: fallbacks = []
        if primary in batch: return batch[primary]
        for alt in fallbacks:
            if alt in batch: return batch[alt]
        return None

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int = 0) -> Dict[str, torch.Tensor]:
        return self._forward_training(batch, epoch=epoch)

    def _forward_training(self, batch: Dict[str, torch.Tensor], epoch: int = 0) -> Dict[str, torch.Tensor]:
        local_graph = self._get_batch_field(batch, DataDict.local_graph)
        vel_val = self._get_batch_field(batch, DataDict.vel)
        c_env, node_embeddings, node_pos = self.perception(local_graph, vel_val)
        start_point_val = batch['traj_gt_local'][:, 0, :]
        c_start = self.start_point_encoder(start_point_val)
        target_normalized = self._get_batch_field(batch, DataDict.target)
        c_target = self.target_encoder(target_normalized)
        gt_traj_normalized = batch['traj_gt']
        
        if self.generator_type == GeneratorType.diffusion:
            batch_indices = getattr(local_graph, 'batch', None)
            final_output = self.diffusion_model.training_loss(
                gt_traj_normalized=gt_traj_normalized,
                gt_traj_absolute=batch['traj_gt_local'],
                scale=batch['traj_scale'],
                c_env=c_env,
                c_target=c_target,
                c_start=c_start,
                true_start_point=batch['traj_gt_local'][:, 0, :],
                node_embeddings=node_embeddings,
                node_pos=node_pos,
                batch_indices=batch_indices,
                epoch=epoch
            )
            return final_output
        else:
            raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int = 1,
        target_conditioning: str = "soft",
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        
        local_graph = self._get_batch_field(batch, DataDict.local_graph)
        vel_val = self._get_batch_field(batch, DataDict.vel)
        
        c_env, node_embeddings, node_pos = self.perception(local_graph, vel_val)
        
        start_point_val = batch['traj_gt_local'][:, 0, :]
        c_start = self.start_point_encoder(start_point_val)
        
        target_normalized = self._get_batch_field(batch, DataDict.target)
        c_target = self.target_encoder(target_normalized)
        
        output = {}
        path_val = self._get_batch_field(batch, DataDict.path, fallbacks=['traj_gt_local'])
        if path_val is not None: output[DataDict.path] = path_val

        if self.generator_type == GeneratorType.diffusion:
            batch_indices = getattr(local_graph, 'batch', None)
            
            # ``num_samples`` is retained for API compatibility; the current
            # sampler still returns one trajectory per input example.
            predicted_traj_normalized = self.diffusion_model.sample(
                c_env=c_env, 
                c_target=c_target,
                c_start=c_start,
                true_start_point=batch['traj_gt_local'][:, 0, :],
                node_embeddings=node_embeddings,
                node_pos=node_pos,
                batch_indices=batch_indices,
                scale=batch['traj_scale'],
                target_conditioning=target_conditioning,
                hard_target_point=target_normalized if target_conditioning == "hard" else None,
            )

            scale = batch['traj_scale']
            shift = batch['traj_shift']
            prediction_local = predicted_traj_normalized * scale.unsqueeze(1) + shift.unsqueeze(1)
            generator_output = {DataDict.prediction: prediction_local}
        else:
            raise NotImplementedError

        output.update(generator_output)
    
        return output
