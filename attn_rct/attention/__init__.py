"""The intervention registry.

Registers all attention arms at import time to ensure consistency. 
The `cfg.attention` configuration dictates which arm is used for a given run.
"""

from .base import AttentionBase, resolve_compute_dtype
from .flash import FlashAttention
from .linear import LinearAttention
from .linformer import LinformerAttention
from .sparse import SparseAttention
from .vanilla import VanillaAttention

ATTENTION_REGISTRY = {
    "vanilla": VanillaAttention,
    "flash": FlashAttention,
    "linformer": LinformerAttention,
    "linear": LinearAttention,
    "sparse": SparseAttention,
}


def build_attention(cfg):
    """Constructs and returns the attention arm specified by cfg.attention.

    Raises:
        KeyError: If the requested attention arm is not registered.
    """
    if cfg.attention not in ATTENTION_REGISTRY:
        raise KeyError(
            f"unknown attention arm {cfg.attention!r}; "
            f"registered: {sorted(ATTENTION_REGISTRY)}"
        )
    return ATTENTION_REGISTRY[cfg.attention](cfg)


__all__ = [
    "ATTENTION_REGISTRY", "build_attention", "AttentionBase", "resolve_compute_dtype",
    "VanillaAttention", "FlashAttention", "LinformerAttention", "LinearAttention",
    "SparseAttention",
]