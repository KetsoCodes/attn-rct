"""Linear Transformer (Katharopoulos et al. 2020).

Uses kernel-based linear attention instead of standard softmax attention.
By changing the similarity function so that it can be factorised, we can use the 
associative property of matrix multiplication. This allows us to compute a shared 
matrix for all queries, reducing the computational complexity from O(N^2) to O(N). 

This acts as a fundamentally different similarity function rather than just an approximation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from .base import AttentionBase

EPS = 1e-6


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """Applies the feature map phi(x) = elu(x) + 1."""
    return F.elu(x) + 1


class LinearAttention(AttentionBase):
    """Kernel-based linear attention implementation."""

    def _attend(self, q, k, v, mask):
        """Computes attention in O(N) time."""
        q = elu_feature_map(q)
        k = elu_feature_map(k)

        if mask is not None:
            keep = mask[:, None, :, None].to(k.dtype)
            k = k * keep

        kv = torch.einsum("bhsd,bhsm->bhdm", k, v)

        k_sum = k.sum(dim=2)
        denominator = torch.einsum("bhsd,bhd->bhs", q, k_sum)
        denominator = 1.0 / (denominator + EPS)

        return torch.einsum("bhsd,bhdm,bhs->bhsm", q, kv, denominator)