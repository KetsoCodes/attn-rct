"""Linformer (Wang et al. 2020) with low-rank projection along the sequence dimension.

The attention matrix is empirically low-rank. Rather than computing N x N, Linformer
projects keys and values from N positions down to K using a learned matrix E, then
attends over those K positions. The result is N x K, giving O(N) complexity when K is fixed.

Note that this reduces the length dimension, not the feature dimension. Projecting along
length mixes information across positions. This is why Linformer cannot support causal
masking efficiently, as a projected position blends past and future. This is perfectly fine
for LRA ListOps since it is an encoder-style classification task.

Parameter cost under layerwise sharing is fixed (K * max_len), independent of depth.
One matrix serves every layer, every head, and both key and value.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from .base import AttentionBase


class SequenceProjection(nn.Module):
    """The shared projection matrix E of shape (k, max_len)."""

    def __init__(self, k: int, max_len: int, seed: int = 0):
        """Initialises the projection matrix E using a dedicated random number generator.

        This prevents E from consuming from the global random stream, ensuring the 
        shared qkv/out projections remain exactly paired with other experimental arms 
        at initialisation.
        """
        super().__init__()
        generator = torch.Generator().manual_seed(seed + 90210)
        weight = torch.empty(k, max_len)
        
        # Manual xavier_uniform (gain 1/sqrt(2)) using our dedicated generator 
        # to avoid touching the global stream.
        gain = 1.0 / math.sqrt(2)
        bound = gain * math.sqrt(6.0 / (k + max_len))
        weight.uniform_(-bound, bound, generator=generator)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Contracts the sequence axis from seq_len down to k."""
        projection = self.weight[:, :seq_len].to(x.dtype)
        return torch.einsum("ks,bhsd->bhkd", projection, x)


def get_shared_projection(cfg) -> SequenceProjection:
    """Returns the single projection matrix for this model, cached on the config to enable layerwise sharing."""
    shared = getattr(cfg, "_linformer_shared", None)
    if shared is None:
        shared = SequenceProjection(
            k=int(cfg.linformer_k),
            max_len=int(cfg.max_len),
            seed=int(getattr(cfg, "seed", 0)),
        )
        cfg._linformer_shared = shared
    return shared


class LinformerAttention(AttentionBase):
    """Low-rank sequence projection attention implementation."""

    def __init__(self, cfg):
        super().__init__(cfg)          # Shared qkv/out projections created first
        self.k = int(cfg.linformer_k)
        self.sharing = getattr(cfg, "linformer_sharing", "layerwise")
        seed = int(getattr(cfg, "seed", 0))

        if self.sharing == "layerwise":
            # One E across all layers, all heads, and both key and value.
            self.key_projection = get_shared_projection(cfg)
            self.value_projection = self.key_projection
        elif self.sharing == "key_value":
            # One E per layer, shared between key and value.
            shared = SequenceProjection(self.k, int(cfg.max_len), seed)
            self.key_projection = shared
            self.value_projection = shared
        elif self.sharing == "headwise":
            # Separate E and F per layer, each shared across heads.
            self.key_projection = SequenceProjection(self.k, int(cfg.max_len), seed)
            self.value_projection = SequenceProjection(self.k, int(cfg.max_len), seed + 1)
        else:
            raise ValueError(
                f"unknown linformer_sharing {self.sharing!r}; "
                "expected layerwise | key_value | headwise"
            )

    def _attend(self, q, k, v, mask):
        """Compresses keys and values to k positions, then attends over them.

        Raises:
            ValueError: If the sequence is longer than the projection matrix E can cover.
        """
        seq_len = q.shape[2]
        if seq_len > self.key_projection.weight.shape[1]:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len "
                f"{self.key_projection.weight.shape[1]}; E cannot be sliced to fit"
            )

        if mask is not None:
            # Zero out padded positions BEFORE projecting. Since E mixes across positions, 
            # an unmasked padded key would leak into all k projected positions where it 
            # could no longer be isolated.
            keep = mask[:, None, :, None].to(k.dtype)
            k = k * keep
            v = v * keep

        k_projected = self.key_projection(k, seq_len)      # (B, H, k, Dh)
        v_projected = self.value_projection(v, seq_len)    # (B, H, k, Dh)

        # No attention mask is applied here because projected positions are blends of tokens. 
        # Padding was already handled prior to projection.
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k_projected.transpose(-2, -1)) * scale   # (B, H, S, k)
        weights = scores.softmax(dim=-1)
        return weights @ v_projected