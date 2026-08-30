"""Vanilla scaled dot-product attention (Vaswani et al. 2017).

This serves as the study's reference arm rather than its baseline. It is both the
naive implementation the IO-aware baseline is timed against, and the ground truth
the flash arm's exactness is checked against.

The score matrix is materialised once and then reused in place: the scale folds into
the queries before the product, masking overwrites the scores, and the softmax reuses
the same buffer. The attention computed is unchanged -- what changes is that we no
longer allocate four (B, H, S, S) tensors per layer to produce one.
"""

from __future__ import annotations

import math
import torch
from .base import AttentionBase, softmax_


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
        # Scaling the queries first is algebraically identical to scaling the product,
        # and avoids a second (B, H, S, S) allocation.
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-2, -1))  # (B, H, S, S) with O(S^2) cost

        if mask is not None:
            # Mask key positions only so a padded token is never attended to.
            # Padded queries still produce outputs that the frame's mean-pool drops.
            scores.masked_fill_(~mask[:, None, None, :], torch.finfo(scores.dtype).min)

        return softmax_(scores, dim=-1) @ v
