"""Vanilla scaled dot-product attention (Vaswani et al. 2017).

This serves as the study's reference arm rather than its baseline. It is both the
naive implementation the IO-aware baseline is timed against, and the ground truth
the flash arm's exactness is checked against.
"""

from __future__ import annotations

import math
import torch
from .base import AttentionBase


class VanillaAttention(AttentionBase):
    """Standard attention implementation materialising the full (B, H, S, S) score matrix."""

    def _attend(self, q, k, v, mask):
        """Computes softmax(QK^T / sqrt(d_k)) V.

        Args:
            q, k, v: (B, H, S, head_dim) projected queries, keys, and values.
            mask: (B, S) key-padding mask where True marks a real token, or None.

        Returns:
            (B, H, S, head_dim) attended values.
        """
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, S, S) with O(S^2) cost

        if mask is not None:
            # Mask key positions only so a padded token is never attended to.
            # Padded queries still produce outputs that the frame's mean-pool drops.
            keep = mask[:, None, None, :]
            scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)

        weights = scores.softmax(dim=-1)
        return weights @ v