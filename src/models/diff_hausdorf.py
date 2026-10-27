"""
HausdorffLoss for DRIFT

Computes a symmetric Hausdorff-like distance between two point sets for each batch sample.

Supports two modes (controlled by src.utils.config.Hausdorff):
 - average:    mean(min_distances(set1->set2)) + mean(min_distances(set2->set1))
 - max:        max(min_distances(set1->set2))  + max(min_distances(set2->set1))

Return:
 - Tensor of shape (B,) containing per-sample distances. If input batch size is 1,
   the returned tensor will be squeezed to a scalar (for backward compatibility).
"""

from typing import Union

import torch
from torch import nn

from src.utils.config import Hausdorff


class HausdorffLoss(nn.Module):
    def __init__(self, mode: Union[Hausdorff, str] = Hausdorff.average):
        """
        Args:
            mode: Hausdorff.average or Hausdorff.max (or corresponding string 'avg'/'max')
        """
        super(HausdorffLoss, self).__init__()
        # Accept either enum value or string
        if isinstance(mode, str):
            m = mode.lower()
            if m in ("avg", "average", "mean"):
                self.mode = Hausdorff.average
            elif m in ("max", "maximum"):
                self.mode = Hausdorff.max
            else:
                raise ValueError(f"Unknown Hausdorff mode string: {mode}")
        else:
            self.mode = mode

    def forward(self, set1: torch.Tensor, set2: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric Hausdorff-like distance between set1 and set2.

        Args:
            set1: Tensor of shape (B, M, C) or (M, C)
            set2: Tensor of shape (B, N, C) or (N, C)

        Returns:
            Tensor of shape (B,) containing distances per-batch.
            If input is single-sample (both 2D), the returned tensor will be squeezed to scalar.
        """
        # Expand single-sample inputs to batch dimension
        if set1.dim() == 2:
            set1 = set1.unsqueeze(0)
        if set2.dim() == 2:
            set2 = set2.unsqueeze(0)

        # Validate shapes
        if set1.dim() != 3 or set2.dim() != 3:
            raise ValueError("Inputs must be 2D (M,C) or 3D (B,M,C) tensors")
        if set1.size(0) != set2.size(0):
            raise AssertionError("Batch size must match between set1 and set2")
        if set1.size(2) != set2.size(2):
            raise AssertionError("Point dimension C must match between set1 and set2")

        B, M, C = set1.shape
        _, N, _ = set2.shape

        # Compute pairwise distances: result shape (B, M, N)
        # Expand dims for broadcasting: set1 -> (B, M, 1, C), set2 -> (B, 1, N, C)
        s1 = set1.unsqueeze(2)  # (B, M, 1, C)
        s2 = set2.unsqueeze(1)  # (B, 1, N, C)
        dists = torch.norm(s1 - s2, dim=-1, p=2)  # (B, M, N)

        # For each point in set1, find distance to closest point in set2: (B, M)
        min_1_to_2, _ = torch.min(dists, dim=2)
        # For each point in set2, find distance to closest point in set1: (B, N)
        min_2_to_1, _ = torch.min(dists, dim=1)

        if self.mode == Hausdorff.average:
            term_1 = torch.mean(min_1_to_2, dim=1)  # (B,)
            term_2 = torch.mean(min_2_to_1, dim=1)  # (B,)
        elif self.mode == Hausdorff.max:
            term_1 = torch.max(min_1_to_2, dim=1)[0]  # (B,)
            term_2 = torch.max(min_2_to_1, dim=1)[0]  # (B,)
        else:
            raise ValueError(f"Unsupported Hausdorff mode: {self.mode}")

        res = term_1 + term_2  # (B,)

        # Keep backward compatibility with earlier code: if batch size is 1, squeeze to scalar
        if res.numel() == 1:
            return res.squeeze(0)
        return res
