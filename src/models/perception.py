# Perception and state encoders used by the trajectory model.

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Tuple

from src.models.backbones.perception_gnn import PerceptionGNN

class VelocityEncoder(nn.Module):
    """Encode a velocity sequence."""
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, out_dim: int = 32, encoder_type: str = 'cnn'):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        if self.encoder_type == 'cnn':
            self.encoder = nn.Sequential(
                nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_dim, out_dim)
            )
        elif self.encoder_type == 'lstm':
            self.lstm = nn.LSTM(1, hidden_dim, batch_first=True, num_layers=2)
            self.fc = nn.Linear(hidden_dim, out_dim)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
    def forward(self, vel: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == 'cnn':
            output = self.encoder(vel.unsqueeze(1))
        elif self.encoder_type == 'lstm':
            _, (h_n, _) = self.lstm(vel.unsqueeze(-1))
            output = self.fc(h_n[-1])
        else:
            output = self.encoder(vel)
        return output

class TargetEncoder(nn.Module):
    """
    Encode target point coordinates for goal conditioning.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        return self.encoder(target)
        
class StartPointEncoder(nn.Module):
    """
    Encode starting point coordinates as a feature vector.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, start_point: torch.Tensor) -> torch.Tensor:
        return self.encoder(start_point)

class PerceptionEncoder(nn.Module):
    """
    Encode the environment point cloud and velocity into ``c_env``.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.perception_gnn = PerceptionGNN(
            in_feats=cfg.gnn_cfg.get('in_feats', 4),
            hidden_dims=cfg.gnn_cfg.get('hidden_dims', [64, 128, 256, 128]),
            out_dim=cfg.gnn_cfg.get('out_dim', 128),
            pool_method=cfg.gnn_cfg.get('pool_method', 'max')
        )
        self.vel_encoder = VelocityEncoder(
            input_dim=cfg.vel_cfg.get('input_dim', 5),
            hidden_dim=cfg.vel_cfg.get('hidden_dim', 64),
            out_dim=cfg.vel_cfg.get('out_dim', 32)
        )
        
        # Fuse environment and velocity features.
        lidar_dim = cfg.gnn_cfg.get('out_dim', 128)
        vel_dim = cfg.vel_cfg.get('out_dim', 32)
        fusion_input_dim = lidar_dim + vel_dim
        
        self.c_env_dim = cfg.fusion_cfg.get('out_dim', 128)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, self.c_env_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.c_env_dim, self.c_env_dim),
            nn.LayerNorm(self.c_env_dim)
        )

    def forward(self, batch_graphs: Batch, vel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode the environment.
        batch_node_embeddings, c_lidar = self.perception_gnn(batch_graphs)
        
        # Encode the velocity.
        c_vel = self.vel_encoder(vel)
        
        # Fuse the inputs into an environment context.
        fusion_input = torch.cat([c_lidar, c_vel], dim=-1)
        c_env = self.fusion_mlp(fusion_input)
        
        return c_env, batch_node_embeddings, batch_graphs.pos
