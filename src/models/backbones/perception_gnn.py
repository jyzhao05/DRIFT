"""
DRIFT/src/models/backbones/perception_gnn.py
PerceptionGNN - GNN backbone for point cloud encoding in CMAGNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, GATConv, global_max_pool, global_mean_pool
from torch_geometric.data import Data
from typing import List, Tuple, Optional
import math


class EdgeConvBlock(nn.Module):
    """EdgeConv block with a residual connection and layer normalization."""

    def __init__(self, in_dim: int, out_dim: int, k: int = 8):
        super().__init__()
        self.k = k
        self.edge_conv = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_dim, out_dim),
                # nn.BatchNorm1d(out_dim),
                nn.LayerNorm(out_dim), 
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim),
                nn.LayerNorm(out_dim) 
                # nn.BatchNorm1d(out_dim)
            ),
            aggr='max'
        )

        # Projection used by the residual connection.
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features with shape (N, in_dim).
            edge_index: Edge indices with shape (2, E).
        Returns:
            torch.Tensor: Updated node features with shape (N, out_dim).
        """
        residual = self.residual(x)
        out = self.edge_conv(x, edge_index)
        return self.activation(out + residual)


class GATBlock(nn.Module):
    """Graph attention block with multiple attention heads."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.gat(x, edge_index)
        return self.activation(self.norm(out + residual))


class PerceptionGNN(nn.Module):
    """
    GNN backbone for encoding local point cloud geometry into structured representations
    Supports EdgeConv (DGCNN-style) and GAT architectures
    """

    def __init__(
        self,
        in_feats: int = 4,
        hidden_dims: List[int] = [64, 128, 256, 128],
        out_dim: int = 128,
        gnn_type: str = 'edgeconv',
        k_nn: int = 12,
        heads: int = 4,
        dropout: float = 0.1,
        pool_method: str = 'max'
    ):
        """
        Initialize PerceptionGNN

        Args:
            in_feats: Input node feature dimension (typically x, y, z, and density).
            hidden_dims: Hidden layer dimensions.
            out_dim: Final node embedding dimension.
            gnn_type: GNN type ('edgeconv' or 'gat').
            k_nn: k-NN graph construction parameter.
            heads: Number of GAT attention heads.
            dropout: Dropout rate.
            pool_method: Global pooling method ('max', 'mean', or 'add').
        """
        super().__init__()

        self.in_feats = in_feats
        self.out_dim = out_dim
        self.gnn_type = gnn_type.lower()
        self.k_nn = k_nn
        self.pool_method = pool_method

        # Build GNN layers.
        dims = [in_feats] + hidden_dims + [out_dim]
        self.gnn_layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            in_dim, out_dim_layer = dims[i], dims[i + 1]

            if self.gnn_type == 'edgeconv':
                layer = EdgeConvBlock(in_dim, out_dim_layer, k=k_nn)
            elif self.gnn_type == 'gat':
                layer = GATBlock(in_dim, out_dim_layer, heads=heads, dropout=dropout)
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")

            self.gnn_layers.append(layer)

        # Global feature projection head.
        self.global_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _edgeconv_block(self, x: torch.Tensor, edge_index: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Internal helper for EdgeConv layers.

        Args:
            x: Node features with shape (N, D).
            edge_index: Edge indices with shape (2, E).
            layer_idx: Layer index.

        Returns:
            torch.Tensor: Updated node features.
        """
        return self.gnn_layers[layer_idx](x, edge_index)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through PerceptionGNN

        Args:
            data: PyG Data object containing:
                - x: node features (N, in_feats)
                - edge_index: edge connectivity (2, E)
                - pos: node positions (N, 3) [optional, for geometric features]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - node_embeddings: Node embeddings with shape (N, out_dim).
                - pooled: Pooled global feature with shape (out_dim,).
        """
        x = data.x  # (N, in_feats)
        edge_index = data.edge_index  # (2, E)
        batch = getattr(data, 'batch', None)

        # Validate input dimensions.
        if x.size(1) != self.in_feats:
            raise ValueError(f"Expected input features dim {self.in_feats}, got {x.size(1)}")

        # Propagate through each layer.
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)

            # Optionally apply dropout between layers.
            if i < len(self.gnn_layers) - 1 and self.training:
                x = F.dropout(x, p=0.1, training=self.training)

        node_embeddings = x  # (N, out_dim)

        # Global pooling.
        if self.pool_method == 'max':
            pooled = global_max_pool(node_embeddings, batch)  # (batch_size, out_dim)
        elif self.pool_method == 'mean':
            pooled = global_mean_pool(node_embeddings, batch)  # (batch_size, out_dim)
        elif self.pool_method == 'add':
            from torch_geometric.nn import global_add_pool
            pooled = global_add_pool(node_embeddings, batch)  # (batch_size, out_dim)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pool_method}")

        # The batch dimension is retained for consistency with batched graphs.
        # if batch is None:
        #     pooled = pooled.squeeze(0)  # (out_dim,)

        # Post-process the global feature.
        pooled = self.global_head(pooled)

        return node_embeddings, pooled

    def get_receptive_field_info(self) -> dict:
        """Return receptive-field information."""
        return {
            'num_layers': len(self.gnn_layers),
            'k_nn': self.k_nn,
            'theoretical_receptive_field': self.k_nn ** len(self.gnn_layers),
            'gnn_type': self.gnn_type
        }

    @torch.jit.ignore
    def visualize_attention(self, data: Data, layer_idx: int = -1) -> Optional[torch.Tensor]:
        """
        Visualize attention weights (GAT only).

        Args:
            data: Input graph data.
            layer_idx: Layer index to inspect.

        Returns:
            torch.Tensor: Attention weight matrix, when available.
        """
        if self.gnn_type != 'gat':
            return None

        # The GAT layer must be configured to return attention weights.
        return None
