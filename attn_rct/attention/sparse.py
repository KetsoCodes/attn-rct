"""Sparse attention patterns (Child et al. 2019 and Beltagy et al. 2020).

Each query attends to a chosen subset of keys instead of all N. We apply the patterns
as a mask over a fully materialised N x N score matrix. The attention we compute is
exactly the paper's so quality results are reportable, but memory and wall-clock
are those of dense attention.

Two things are deliberate about how the mask is applied. The pattern is (H, S, S) and
broadcasts over the batch, and the padding mask is (B, 1, 1, S) and broadcasts over
heads and queries, so neither is ever expanded to (B, H, S, S) -- combining them into a
single boolean would cost more memory than the scores themselves. And the pattern is
built once at max_len and sliced, so the cache holds one tensor per device rather than
one per distinct padded batch length.
"""

from __future__ import annotations

import math
import torch
from .base import AttentionBase, softmax_


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
        self.max_len = int(cfg.max_len)
        # 0 or None means auto. We resolve it against max_len, not against the padded
        # length of whatever batch arrives first: with dynamic padding the latter makes
        # the attention pattern a function of batch composition, which is a nuisance
        # variable in the intervention and leaves the mask cache unbounded.
        configured = getattr(cfg, "sparse_stride", None)
        self.stride = int(configured) if configured else max(1, round(math.sqrt(self.max_len)))
        self._full: dict = {}       # device -> (allowed, blocked) at max_len
        self._cache: dict = {}      # (seq_len, device) -> (H, S, S) bool view of allowed
        self._blocked: dict = {}    # (seq_len, device) -> (H, S, S) bool view of ~allowed

    def _build(self, device):
        """Builds the (H, max_len, max_len) pattern once per device, with its negation.

        We keep the negation alongside it because masked_fill_ fills where the mask is
        True, so the hot path needs "blocked" and the accessor needs "allowed", and
        recomputing either per call would allocate a full pattern every batch.
        """
        key = str(device)
        if key in self._full:
            return self._full[key]

        builder = PATTERNS[self.pattern_name]
        allowed = torch.stack(
            [builder(self.max_len, self.stride, head) for head in range(self.n_heads)]
        ).to(device)

        # Force the diagonal to ensure every query keeps at least one key, preventing NaN softmax.
        diagonal = torch.eye(self.max_len, dtype=torch.bool, device=device)
        allowed = allowed | diagonal[None]

        self._full[key] = (allowed, ~allowed)
        return self._full[key]

    def pattern_mask(self, seq_len: int, device) -> torch.Tensor:
        """Returns the cached per-head boolean attention pattern, True where attending is allowed."""
        return self._masks(seq_len, device)[0]

    def _masks(self, seq_len: int, device):
        """Returns (allowed, blocked) views for this length, sliced from the max_len pattern.

        Raises:
            ValueError: If the sequence is longer than the pattern was built for.
        """
        key = (seq_len, str(device))
        if key in self._cache:
            return self._cache[key], self._blocked[key]

        if seq_len > self.max_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len {self.max_len}; "
                "the pattern cannot be sliced to fit"
            )

        allowed, blocked = self._build(device)
        # Views, not copies: every pattern here depends only on row and column indices,
        # so the top-left corner of the max_len mask is the mask for a shorter sequence.
        self._cache[key] = allowed[:, :seq_len, :seq_len]
        self._blocked[key] = blocked[:, :seq_len, :seq_len]
        return self._cache[key], self._blocked[key]

    def _attend(self, q, k, v, mask):
        """Applies the configured sparse pattern and key-padding mask before computing attention."""
        seq_len = q.shape[2]
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-2, -1))   # (B, H, S, S) materialises full matrix
        blocked_value = torch.finfo(scores.dtype).min

        _, blocked = self._masks(seq_len, q.device)
        scores.masked_fill_(blocked[None], blocked_value)       # (H, S, S) broadcast over the batch

        if mask is not None:
            scores.masked_fill_(~mask[:, None, None, :], blocked_value)

            # A row is empty exactly when its query is padding: the pattern always keeps
            # the diagonal, so a real query always retains at least its own real key.
            # Hand those rows key 0 back, as the previous implementation did, so padded
            # outputs stay one-hot rather than becoming a uniform average. The frame's
            # mean-pool discards them regardless.
            empty_rows = scores.amax(dim=-1) == blocked_value   # (B, H, S)
            scores[..., 0].masked_fill_(empty_rows, 0.0)

        return softmax_(scores, dim=-1) @ v
