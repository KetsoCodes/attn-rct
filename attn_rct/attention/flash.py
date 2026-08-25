"""FlashAttention-2 (Dao et al. 2022; Dao 2023) baseline implementation.

This is an exact attention implementation that computes the same function as standard 
attention, but minimises memory traffic to achieve speedups. It establishes the baseline 
efficiency that the approximate arms must outperform.

Includes two execution paths: a true FA-2 path using variable-length unpadding, 
and a fallback PyTorch SDPA path. The require_flash flag ensures we never silently 
measure the fallback kernel when evaluating efficiency.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from .base import AttentionBase

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    HAS_FLASH_LIB = True
except ImportError:
    HAS_FLASH_LIB = False


def _unpad(tensor, mask):
    """Calls unpad_input, indexing the result to support varying return lengths across flash-attn versions."""
    out = unpad_input(tensor, mask)
    return out[0], out[1], out[2], out[3]


class FlashAttention(AttentionBase):
    """FlashAttention-2 baseline.

    Config:
        require_flash: If True, raises an error instead of falling back to a non-FA-2 kernel.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.require_flash = bool(getattr(cfg, "require_flash", False))
        self._path_logged = None  

    @staticmethod
    def measure_backend(device: torch.device) -> str:
        """Runs a test attention call to determine which backend is actually reachable on the device."""
        if device.type != "cuda":
            return "math (CPU) - NOT FlashAttention"

        major, minor = torch.cuda.get_device_capability(device)
        arch = f"sm_{major}{minor}"

        if HAS_FLASH_LIB and major >= 8:
            try:
                q = torch.randn(4, 2, 8, device=device, dtype=torch.bfloat16)
                cu = torch.tensor([0, 4], device=device, dtype=torch.int32)
                flash_attn_varlen_func(q, q, q, cu, cu, 4, 4, causal=False)
                return f"flash-attn library varlen FA-2 ({arch}) - VERIFIED"
            except Exception as err:
                return f"flash-attn present but call failed on {arch}: {err!r}"

        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            q = torch.randn(1, 2, 8, 16, device=device, dtype=torch.bfloat16)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                F.scaled_dot_product_attention(q, q, q)
            return (f"torch SDPA FLASH_ATTENTION ({arch}) - VERIFIED, "
                    "but only reachable with attn_mask=None")
        except Exception as err:
            return f"FA-2 unavailable on {arch}: {type(err).__name__}: {err}"

    def _attend_varlen(self, q, k, v, mask):
        """Executes true FlashAttention-2 by unpadding the batch to avoid needing an attention mask."""
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        batch, seq_len = q.shape[0], q.shape[1]

        q_flat, indices, cu_seqlens, max_s = _unpad(q, mask)
        k_flat, _, _, _ = _unpad(k, mask)
        v_flat, _, _, _ = _unpad(v, mask)

        context_flat = flash_attn_varlen_func(
            q_flat, k_flat, v_flat,
            cu_seqlens, cu_seqlens, max_s, max_s,
            dropout_p=0.0,          
            causal=False,           
        )

        context = pad_input(context_flat, indices, batch, seq_len)
        return context.transpose(1, 2)

    def _attend_sdpa(self, q, k, v, mask):
        """Fallback path using PyTorch SDPA with an explicit mask. Exact, but not true FA-2."""
        attn_mask = mask[:, None, None, :] if mask is not None else None
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)

    def _attend(self, q, k, v, mask):
        """Routes to the true FA-2 varlen path if possible, otherwise falls back or raises an error."""
        can_varlen = (
            HAS_FLASH_LIB
            and q.is_cuda
            and mask is not None
            and q.dtype in (torch.float16, torch.bfloat16)
        )

        if can_varlen:
            self._path_logged = "flash_varlen_fa2"
            return self._attend_varlen(q, k, v, mask)

        if self.require_flash:
            raise RuntimeError(
                "require_flash=True but the true FA-2 path is unreachable "
                f"(flash_attn installed={HAS_FLASH_LIB}, cuda={q.is_cuda}, "
                f"dtype={q.dtype}, mask={'yes' if mask is not None else 'no'}). "
                "Refusing to silently measure a different kernel. "
                f"Backend probe says: {self.measure_backend(q.device)}"
            )

        self._path_logged = "sdpa_fallback_NOT_fa2"
        return self._attend_sdpa(q, k, v, mask)