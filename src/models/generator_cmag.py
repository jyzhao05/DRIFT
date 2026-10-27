# CMAGNN trajectory generator with sparse environment attention.

import torch
import torch.nn as nn
from typing import Optional

from .backbones.gt_gru import GTGRUCell 
from .backbones.unet import SinusoidalPosEmb
from torch_geometric.utils import to_dense_batch
from torch_cluster import knn

class SparseCrossAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, value_dim: int, num_heads: int, k_env: int):
        super().__init__()
        assert query_dim % num_heads == 0, "Query dim must be divisible by num_heads"
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.k_env = k_env
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(key_dim, query_dim, bias=False)
        self.to_v = nn.Linear(value_dim, query_dim, bias=False)
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_k = nn.LayerNorm(query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)
        # Optional diagnostics hook. It is disabled by default and therefore
        # has no effect on training/evaluation unless an analysis script sets
        # `capture_debug = True`.
        self.capture_debug = False
        self.debug_records = []
        self.debug_context = {}

    def forward(self, query_features: torch.Tensor, query_pos: torch.Tensor, key_pos: torch.Tensor, key_embed: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, M, _ = query_features.shape
        B_k, N, _ = key_pos.shape
        device = query_features.device
        assert B == B_k, "Batch size mismatch"

        key_pos_2d = key_pos[..., :2]
        key_pos_flat = key_pos_2d.reshape(-1, 2).to(query_features.dtype)
        query_pos_flat = query_pos.reshape(-1, 2)
        key_batch = torch.arange(B, device=device).repeat_interleave(N)
        query_batch = torch.arange(B, device=device).repeat_interleave(M)

        _, global_indices_flat = knn(key_pos_flat, query_pos_flat, self.k_env, batch_x=key_batch, batch_y=query_batch)
        global_indices = global_indices_flat.view(B, M, self.k_env)
        
        batch_offsets = torch.arange(B, device=device).view(B, 1, 1) * N
        local_indices = torch.clamp(global_indices - batch_offsets, 0, N - 1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, M, self.k_env)
        
        gathered_k_embed = key_embed[batch_idx, local_indices]

        q = self.norm_q(self.to_q(query_features))
        k = self.norm_k(self.to_k(gathered_k_embed))
        v = self.to_v(gathered_k_embed)

        q = q.view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, M, self.k_env, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = v.view(B, M, self.k_env, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        q = q.unsqueeze(3)
        dots = (q @ k.transpose(-1, -2)).squeeze(3) * self.scale
        
        if mask is not None:
            gathered_mask = torch.gather(mask.unsqueeze(1).expand(-1, M, -1), 2, local_indices)
            dots.masked_fill_(~gathered_mask.unsqueeze(1), float('-inf'))
        
        attn = dots.softmax(dim=-1)
        if self.capture_debug:
            with torch.no_grad():
                gathered_key_pos = key_pos_2d[batch_idx, local_indices]
                query_pos_dbg = query_pos.unsqueeze(2)
                distances = torch.linalg.norm(gathered_key_pos - query_pos_dbg, dim=-1)
                attn_mean = attn.mean(dim=1)
                entropy = -(attn_mean.clamp_min(1e-12) * attn_mean.clamp_min(1e-12).log()).sum(dim=-1)
                self.debug_records.append({
                    **getattr(self, "debug_context", {}),
                    "local_indices": local_indices.detach().cpu().tolist(),
                    "attention_mean": attn_mean.detach().cpu().tolist(),
                    "attention_entropy": entropy.detach().cpu().tolist(),
                    "neighbor_distances": distances.detach().cpu().tolist(),
                    "query_pos": query_pos.detach().cpu().tolist(),
                    "neighbor_pos": gathered_key_pos.detach().cpu().tolist(),
                })
        out = (attn.unsqueeze(3) @ v).squeeze(3)
        out = out.permute(0, 2, 1, 3).reshape(B, M, self.query_dim)
        
        return self.to_out(out)

class CMAGNNGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        generator_cfg = cfg.generator
        perception_cfg = cfg.perception 

        self.cfg = generator_cfg
        self.conditioning_mode = getattr(generator_cfg, 'conditioning_mode', 'dynamic')
        if self.conditioning_mode not in {'dynamic', 'static', 'pooled_scene', 'pooled_node_global'}:
            raise ValueError(f"Unsupported conditioning_mode: {self.conditioning_mode}")
        self.denoiser_cell = getattr(generator_cfg, 'denoiser_cell', 'gtgru')
        if self.denoiser_cell not in {'gtgru', 'plain_gru'}:
            raise ValueError(f"Unsupported denoiser_cell: {self.denoiser_cell}")
        
        hidden_dim = getattr(generator_cfg, 'hidden_dim', 256)
        time_emb_dim = getattr(generator_cfg, 'time_emb_dim', 128)
        
        c_env_dim = getattr(perception_cfg, 'output_dim', 128)
        c_target_dim = getattr(perception_cfg.target_cfg, 'out_dim', 32)
        node_embed_dim = getattr(perception_cfg, 'gnn_output_dim', 128)
        c_start_dim = 32

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.point_embed = nn.Linear(2, hidden_dim)

        if self.conditioning_mode == 'dynamic':
            self.cross_attention = SparseCrossAttention(
                query_dim=hidden_dim,
                key_dim=node_embed_dim,
                value_dim=node_embed_dim,
                num_heads=getattr(generator_cfg, 'attention_heads', 4),
                k_env=getattr(generator_cfg, 'k_env', 16)
            )
        if self.conditioning_mode in {'static', 'pooled_node_global'}:
            self.env_embed_proj = nn.Linear(node_embed_dim, hidden_dim)
        if self.conditioning_mode == 'pooled_scene':
            self.env_global_proj = nn.Linear(c_env_dim, hidden_dim)

        if self.denoiser_cell == 'gtgru':
            self.gru_cell = GTGRUCell(
                input_size=hidden_dim, hidden_size=hidden_dim,
                context_size=hidden_dim, time_emb_dim=time_emb_dim
            )
        else:
            self.gru_input_fuser = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim + time_emb_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
            self.gru_cell = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        
        self.target_proj = nn.Linear(c_target_dim, hidden_dim)

        global_context_dim = c_env_dim + c_start_dim + c_target_dim
        if self.conditioning_mode == 'pooled_node_global':
            global_context_dim += hidden_dim
        self.global_context_fuser = nn.Sequential(
            nn.Linear(global_context_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2)
        )

    def _convert_pyg_to_dense(self, batch_node_embeddings, batch_node_pos, batch_indices):
        dense_embeddings, mask = to_dense_batch(batch_node_embeddings, batch_indices)
        dense_pos, _ = to_dense_batch(batch_node_pos, batch_indices)
        return dense_embeddings, dense_pos, mask

    def _masked_mean_context(self, dense_node_embeddings, node_mask):
        mask = node_mask.unsqueeze(-1).to(dense_node_embeddings.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (dense_node_embeddings * mask).sum(dim=1) / denom
        return self.env_embed_proj(pooled)

    def forward(self, noisy_traj, t, c_env, c_target, c_start, true_start_point, node_embeddings, node_pos, batch_indices, scale):
        """Generate a trajectory autoregressively with global context injection."""
        B, M, C = noisy_traj.shape
        device = noisy_traj.device
        dense_node_embeddings = dense_node_pos = node_mask = None
        if self.conditioning_mode != 'pooled_scene':
            dense_node_embeddings, dense_node_pos, node_mask = self._convert_pyg_to_dense(node_embeddings, node_pos, batch_indices)
        
        t_embed = self.time_embed(t)

        node_pos_normalized = None
        if self.conditioning_mode == 'dynamic':
            start_point_expanded = true_start_point.unsqueeze(1).expand(-1, dense_node_pos.size(1), -1)
            node_pos_centered = dense_node_pos[:, :, :2] - start_point_expanded
            scale_expanded = scale.unsqueeze(1).expand(-1, dense_node_pos.size(1), -1)
            node_pos_normalized = node_pos_centered / scale_expanded

        static_context = None
        if self.conditioning_mode in {'static', 'pooled_node_global'}:
            static_context = self._masked_mean_context(dense_node_embeddings, node_mask)
        elif self.conditioning_mode == 'pooled_scene':
            static_context = self.env_global_proj(c_env)

        global_context_parts = [c_env, c_start, c_target]
        if self.conditioning_mode == 'pooled_node_global':
            global_context_parts.append(static_context)
        global_context = torch.cat(global_context_parts, dim=-1)
        h_prev = self.global_context_fuser(global_context)

        target_embedding = self.target_proj(c_target)
        
        start_token = torch.zeros(B, C, device=device)

        outputs = []
        for i in range(M):
            prev_noisy_point = start_token if i == 0 else noisy_traj[:, i - 1, :]
            gru_input = self.point_embed(prev_noisy_point) + target_embedding

            current_noisy_point = noisy_traj[:, i, :]
            
            if self.conditioning_mode in {'static', 'pooled_scene', 'pooled_node_global'}:
                dynamic_context = static_context
            else:
                if getattr(self.cross_attention, "capture_debug", False):
                    call_index = len(getattr(self.cross_attention, "debug_records", []))
                    self.cross_attention.debug_context = {
                        "generator_call_index": int(call_index // max(M, 1)),
                        "waypoint_index": int(i),
                        "timestep": t.detach().cpu().tolist(),
                    }
                dynamic_context = self.cross_attention(
                    query_features=h_prev.unsqueeze(1),
                    query_pos=current_noisy_point.unsqueeze(1),
                    key_pos=node_pos_normalized,
                    key_embed=dense_node_embeddings, 
                    mask=node_mask
                ).squeeze(1)

            if self.denoiser_cell == 'gtgru':
                h_t = self.gru_cell(x=gru_input, h_prev=h_prev, context=dynamic_context, time_embed=t_embed, c_target=c_target)
            else:
                combined_input = torch.cat([gru_input, dynamic_context, t_embed], dim=-1)
                h_t = self.gru_cell(self.gru_input_fuser(combined_input), h_prev)
            
            predicted_point = self.output_head(h_t)
            outputs.append(predicted_point)
            
            h_prev = h_t

        pred = torch.stack(outputs, dim=1)
        return pred
