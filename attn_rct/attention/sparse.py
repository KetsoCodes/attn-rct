"""Sparse attention patterns (Child et al. 2019 and Beltagy et al. 2020).

Each query attends to a chosen subset of keys instead of all N. We apply the patterns 
as a mask over a fully materialised N x N score matrix. The attention we compute is 
exactly the paper's so quality results are reportable, but memory and wall-clock 
are those of dense attention.
"""

from __future__ import annotations

import math
import torch
from .base import AttentionBase


# Pattern builders returning an (S, S) boolean mask where True means "may attend".
def strided_pattern(seq_len: int, stride: int, head: int) -> torch.Tensor:
    """Child et al.'s strided factorisation, made symmetric for encoder use.

    Even heads take the local component (nearest stride positions) and odd heads 
    take the strided component (every stride-th position).
    """
    positions = torch.arange(seq_len)
    row, col = positions[:, None], positions[None, :]

    if head % 2 == 0:
        return (row - col).abs() < stride
    return (row - col).abs() % stride == 0


def fixed_pattern(seq_len: int, stride: int, head: int) -> torch.Tensor:
    """Child et al.'s fixed factorisation, made symmetric.

    Even heads attend within their own block of width stride. Odd heads attend to the 
    summary column ending each block.
    """
    positions = torch.arange(seq_len)
    row, col = positions[:, None], positions[None, :]

    if head % 2 == 0:
        return (row // stride) == (col // stride)
    
    # Explicitly expand to (S, S) since the test depends only on the column.
    return ((col % stride) == (stride - 1)).expand(seq_len, seq_len).clone()


def local_pattern(seq_len: int, window: int, head: int) -> torch.Tensor:
    """Sliding window where each position attends to its nearest neighbours."""
    positions = torch.arange(seq_len)
    return (positions[:, None] - positions[None, :]).abs() <= window // 2


def local_global_pattern(seq_len: int, window: int, head: int,
                         n_global: int = 64) -> torch.Tensor:
    """Longformer sliding window plus a block of global tokens.

    Uses the leading block as global tokens, allowing everyone to attend to them 
    and them to attend to everyone.
    """
    mask = local_pattern(seq_len, window, head)
    mask[:, :n_global] = True     # Everyone attends to the global block
    mask[:n_global, :] = True     # The global block attends to everyone
    return mask


PATTERNS = {
    "strided": strided_pattern,
    "fixed": fixed_pattern,
    "local": local_pattern,
    "local_global": local_global_pattern,
}


def block_density(mask: torch.Tensor, block: int = 128) -> float:
    """Calculates the fraction of block tiles containing at least one allowed entry.

    This metric determines whether a pattern can benefit from a block-sparse kernel.
    """
    seq_len = mask.shape[0]
    n_blocks = math.ceil(seq_len / block)
    occupied = sum(
        1
        for i in range(n_blocks)
        for j in range(n_blocks)
        if mask[i * block:(i + 1) * block, j * block:(j + 1) * block].any()
    )
    return occupied / (n_blocks * n_blocks)


class SparseAttention(AttentionBase):
    """Masked-dense sparse attention implementation."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pattern_name = getattr(cfg, "sparse_pattern", "fixed")
        if self.pattern_name not in PATTERNS:
            raise ValueError(
                f"unknown sparse_pattern {self.pattern_name!r}; "
                f"expected one of {sorted(PATTERNS)}"
            )
        self.stride = getattr(cfg, "sparse_stride", None)
        self._cache: dict = {}      # (seq_len, device) mapping to (H, S, S) bool

    def pattern_mask(self, seq_len: int, device) -> torch.Tensor:
        """Builds and caches the per-head boolean attention pattern."""
        key = (seq_len, str(device))
        if key in self._cache:
            return self._cache[key]

        stride = self.stride or max(1, round(math.sqrt(seq_len)))
        builder = PATTERNS[self.pattern_name]
        mask = torch.stack(
            [builder(seq_len, stride, head) for head in range(self.n_heads)]
        ).to(device)

        # Force the diagonal to ensure every query keeps at least one key, preventing NaN softmax.
        diagonal = torch.eye(seq_len, dtype=torch.bool, device=device)
        mask = mask | diagonal[None]

        self._cache[key] = mask
        return mask

    def _attend(self, q, k, v, mask):
        """Applies the configured sparse pattern and key-padding mask before computing attention."""
        seq_len = q.shape[2]
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale      # (B, H, S, S) materialises full matrix

        allowed = self.pattern_mask(seq_len, q.device)[None]
        if mask is not None:
            allowed = allowed & mask[:, None, None, :]

        # Allow empty padded rows to attend to position 0 to keep softmax finite. 
        # These are safely discarded by the mean-pool later.
        empty_rows = ~allowed.any(dim=-1, keepdim=True)
        if empty_rows.any():
            rescue = torch.zeros_like(allowed)
            rescue[..., 0] = True
            allowed = allowed | (empty_rows & rescue)

        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        return scores.softmax(dim=-1) @ v