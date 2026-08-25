"""Shared contract for every attention arm.

Enforces three invariants to guarantee a strictly paired experimental design:
1. Unified interface: forward(x, mask) -> (B, S, D).
2. Shared parameterisation: Fused QKV and output projections are built here in a fixed 
   order to ensure bit-identical weights across all arms for a given seed.
3. Single precision policy: The attention dtype is set once and applied universally.

Attention-weight dropout is deliberately disabled for all arms.
"""

from __future__ import annotations

from types import SimpleNamespace
import torch
import torch.nn as nn


def resolve_compute_dtype(cfg, device: torch.device) -> torch.dtype:
    """Determines the appropriate compute data type for the attention operation.

    Defaults to bf16 on capable hardware (sm_80+) to avoid loss scaling, 
    and fp32 everywhere else unless explicitly overridden.
    """
    requested = getattr(cfg, "compute_dtype", "auto")
    if requested != "auto":
        return {"fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16}[requested]
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float32


class AttentionBase(nn.Module):
    """Base class for all attention arms.

    Handles projection, head reshaping, precision handling, and output projection to ensure 
    all arms remain structurally identical. Subclasses only need to implement the `_attend` method.
    """

    def __init__(self, cfg):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(
                f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
            )

        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.attn_dropout = float(getattr(cfg, "attn_dropout", 0.0))

        # Shared parameters are initialized first to preserve RNG state across variants.
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self._compute_dtype_cfg = getattr(cfg, "compute_dtype", "auto")

    def compute_dtype(self, device: torch.device) -> torch.dtype:
        """Resolves the attention data type for this specific arm on the given device."""
        return resolve_compute_dtype(
            SimpleNamespace(compute_dtype=self._compute_dtype_cfg), device
        )

    @staticmethod
    def check_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor | None:
        """Validates the padding mask (where 1 is a real token and 0 is padding).

        Raises:
            ValueError: On a shape mismatch, or if any sequence is entirely padding.
        """
        if mask is None:
            return None
        if mask.shape != x.shape[:2]:
            raise ValueError(f"mask shape {tuple(mask.shape)} != {tuple(x.shape[:2])}")
        mask = mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("a sequence in this batch is entirely padding")
        return mask

    def project_qkv(self, x: torch.Tensor):
        """Projects the input and splits it into queries, keys, and values across heads."""
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def merge_heads(self, context: torch.Tensor) -> torch.Tensor:
        """Concatenates attention heads back into the original model dimension."""
        batch, _, seq_len, _ = context.shape
        return context.transpose(1, 2).reshape(batch, seq_len, self.d_model)

    def _attend(self, q, k, v, mask):
        """Core attention mechanism. Implemented by subclasses."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Executes the attention forward pass over a padded batch."""
        mask = self.check_mask(x, mask)
        dtype = self.compute_dtype(x.device)

        q, k, v = self.project_qkv(x)
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)

        context = self._attend(q, k, v, mask)

        context = self.merge_heads(context).to(x.dtype)
        return self.out_proj(context)