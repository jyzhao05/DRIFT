# Auxiliary trajectory losses and stage-specific weighting.

import torch
import torch.nn as nn
from typing import Dict, Any

from src.models.diff_hausdorf import HausdorffLoss
from tslearn.metrics import dtw
import numpy as np


DEFAULT_STAGE_RATIOS = {
    "imitation": {
        "distance_ratio": 20.0,
        "dtw_ratio": 15.0,
        "start_ratio": 10.0,
        "last_ratio": 10.0,
        "velocity_ratio": 1.0,
        "collision_ratio": 5.0,
        "smoothness_ratio": 1.0,
    },
    "planning": {
        "distance_ratio": 15.0,
        "dtw_ratio": 10.0,
        "start_ratio": 10.0,
        "last_ratio": 15.0,
        "velocity_ratio": 0.5,
        "collision_ratio": 15.0,
        "smoothness_ratio": 1.5,
    },
}


ABLATION_PROFILE_OVERRIDES = {
    "full": {},
    "default": {},
    "no_smoothness": {
        "imitation": {"smoothness_ratio": 0.0},
        "planning": {"smoothness_ratio": 0.0},
    },
    "no_collision": {
        "imitation": {"collision_ratio": 0.0},
        "planning": {"collision_ratio": 0.0},
    },
    "simple_only": {
        "imitation": {
            "distance_ratio": 0.0,
            "dtw_ratio": 0.0,
            "start_ratio": 0.0,
            "last_ratio": 0.0,
            "velocity_ratio": 0.0,
            "collision_ratio": 0.0,
            "smoothness_ratio": 0.0,
        },
        "planning": {
            "distance_ratio": 0.0,
            "dtw_ratio": 0.0,
            "start_ratio": 0.0,
            "last_ratio": 0.0,
            "velocity_ratio": 0.0,
            "collision_ratio": 0.0,
            "smoothness_ratio": 0.0,
        },
    },
    "simple_plus_endpoint": {
        "imitation": {
            "distance_ratio": 0.0,
            "dtw_ratio": 0.0,
            "velocity_ratio": 0.0,
            "collision_ratio": 0.0,
            "smoothness_ratio": 0.0,
        },
        "planning": {
            "distance_ratio": 0.0,
            "dtw_ratio": 0.0,
            "velocity_ratio": 0.0,
            "collision_ratio": 0.0,
            "smoothness_ratio": 0.0,
        },
    },
}

class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.training_cfg = cfg.training
        self.loss_cfg = getattr(cfg, "loss", cfg)
        
        self.trajectory_loss_fn = HausdorffLoss(mode="avg")
        self.endpoint_loss_fn = nn.MSELoss(reduction='mean')
        self.velocity_loss_fn = nn.MSELoss(reduction='mean')

        self.collision_radius = float(getattr(self.loss_cfg, "collision_distance", 0.5))
        # Retained for compatibility; the active weight is selected by forward().
        self.collision_weight = 1.0 
        self.ablation_profile = str(getattr(self.loss_cfg, "ablation_profile", "full")).lower()
        self.custom_stage_ratio_overrides = getattr(self.loss_cfg, "stage_ratio_overrides", None)

    def _get_stage_ratios(self, epoch: int):
        stage1_epochs = getattr(self.training_cfg, 'stage1_epochs', 2)
        stage_name = "imitation" if epoch < stage1_epochs else "planning"
        mode = "Imitation" if stage_name == "imitation" else "Planning"

        ratios = DEFAULT_STAGE_RATIOS[stage_name].copy()
        profile_overrides = ABLATION_PROFILE_OVERRIDES.get(self.ablation_profile, {})
        ratios.update(profile_overrides.get(stage_name, {}))

        if self.custom_stage_ratio_overrides:
            stage_overrides = getattr(self.custom_stage_ratio_overrides, stage_name, None)
            if isinstance(stage_overrides, dict):
                ratios.update(stage_overrides)

        return stage_name, mode, ratios

    def compute_collision_loss(self, trajectory, point_cloud, batch_indices):
        if point_cloud is None or point_cloud.shape[0] == 0:
            return torch.tensor(0.0, device=trajectory.device)
        batch_size = trajectory.shape[0]
        collision_losses = []
        for i in range(batch_size):
            traj_i, pc_mask = trajectory[i], (batch_indices == i)
            pc_i = point_cloud[pc_mask][:, :2]
            if pc_i.shape[0] == 0:
                collision_losses.append(torch.tensor(0.0, device=trajectory.device))
                continue
            dists = torch.cdist(traj_i, pc_i)
            min_dists, _ = torch.min(dists, dim=1)
            violations = torch.clamp(self.collision_radius - min_dists, min=0)
            collision_losses.append(torch.sum(violations ** 2))
        return torch.stack(collision_losses).mean()

    def forward(self,
                predicted_x0: torch.Tensor,
                gt_traj_normalized: torch.Tensor,
                gt_traj_absolute: torch.Tensor,
                point_cloud: torch.Tensor,
                batch_indices: torch.Tensor,
                epoch: int = 0,
                **kwargs) -> Dict[str, torch.Tensor]:
        
        loss_info = {}
        stage_name, mode, ratios = self._get_stage_ratios(epoch)
        distance_ratio = ratios["distance_ratio"]
        dtw_ratio = ratios["dtw_ratio"]
        start_ratio = ratios["start_ratio"]
        last_ratio = ratios["last_ratio"]
        velocity_ratio = ratios["velocity_ratio"]
        collision_ratio = ratios["collision_ratio"]
        smoothness_ratio = ratios["smoothness_ratio"]

        # Compute component losses.
        
        trajectory_loss = self.trajectory_loss_fn(predicted_x0, gt_traj_normalized).mean()
        loss_info['L_trajectory_matching'] = trajectory_loss

        endpoint_loss = self.endpoint_loss_fn(predicted_x0[:, -1, :], gt_traj_normalized[:, -1, :])
        loss_info['L_endpoint'] = endpoint_loss
        
        startpoint_loss = self.endpoint_loss_fn(predicted_x0[:, 0, :], gt_traj_normalized[:, 0, :])
        loss_info['L_startpoint'] = startpoint_loss

        pred_velocities = torch.diff(predicted_x0, dim=1)
        gt_velocities = torch.diff(gt_traj_normalized, dim=1)
        velocity_loss = self.velocity_loss_fn(pred_velocities, gt_velocities)
        loss_info['L_velocity_matching'] = velocity_loss

        collision_loss = self.compute_collision_loss(gt_traj_absolute, point_cloud, batch_indices)
        loss_info['L_collision'] = collision_loss
        
        pred_numpy = predicted_x0.detach().cpu().numpy()
        gt_numpy = gt_traj_normalized.detach().cpu().numpy()
        dtw_losses_values = [dtw(pred_numpy[i], gt_numpy[i]) for i in range(pred_numpy.shape[0])]
        dtw_loss = torch.tensor(dtw_losses_values, device=predicted_x0.device, dtype=predicted_x0.dtype).mean()
        dtw_grad_proxy = torch.mean((predicted_x0 - gt_traj_normalized.detach())**2)
        dtw_loss_with_grad = dtw_loss + (dtw_grad_proxy - dtw_grad_proxy.detach())
        loss_info['L_dtw'] = dtw_loss_with_grad

        # Penalize changes in direction between consecutive trajectory segments.
        pred_diffs = torch.diff(predicted_x0, n=1, dim=1)
        # Use a small epsilon to avoid division by zero.
        pred_directions = torch.nn.functional.normalize(pred_diffs, p=2, dim=2, eps=1e-8)
        # A dot product near one indicates a smooth change in direction.
        # A single segment cannot define a smoothness term.
        if pred_directions.shape[1] > 1:
            smoothness_loss = (1 - (pred_directions[:, :-1, :] * pred_directions[:, 1:, :]).sum(dim=2)).mean()
        else:
            smoothness_loss = torch.tensor(0.0, device=predicted_x0.device)
            
        loss_info['L_smoothness'] = smoothness_loss
        # Return the weighted auxiliary loss sum.
        total_aux_loss = (distance_ratio * trajectory_loss +
                          start_ratio * startpoint_loss +
                          last_ratio * endpoint_loss +
                          velocity_ratio * velocity_loss +
                          collision_ratio * collision_loss +
                          dtw_ratio * dtw_loss_with_grad +
                          smoothness_ratio * smoothness_loss)
                      
        loss_info['loss'] = total_aux_loss  # DiffusionModel adds this to L_simple.
        loss_info['debug_mode(0=Imit,1=Plan)'] = torch.tensor(1.0 if mode == "Planning" else 0.0, device=total_aux_loss.device)
        loss_info['debug_loss_profile'] = torch.tensor(
            ["full", "default", "no_smoothness", "no_collision", "simple_only", "simple_plus_endpoint"].index(self.ablation_profile)
            if self.ablation_profile in ["full", "default", "no_smoothness", "no_collision", "simple_only", "simple_plus_endpoint"]
            else -1,
            device=total_aux_loss.device,
        )
        return loss_info
