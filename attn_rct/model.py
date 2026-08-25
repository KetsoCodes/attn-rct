"""The shared transformer model framework.

Everything except the attention mechanism is held constant across experiments.
Embeddings, positional encoding, feed-forward networks, normalization, pooling, 
and the classifier head are identical by construction. We use pre-normalization 
throughout to ensure stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from .attention import build_attention


class Block(nn.Module):
    """A single pre-norm transformer block."""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attention = build_attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask):
        """Applies attention and the feed-forward network using residual connections."""
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerClassifier(nn.Module):
    """Encoder-style sequence classifier for ListOps."""

    def __init__(self, cfg, vocab_size, n_classes=10):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, cfg.d_model, padding_idx=0)
        self.positional = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.d_model))
        nn.init.normal_(self.positional, std=0.02)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, n_classes)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, tokens, mask):
        """Embeds, encodes, pools, and classifies the token sequence."""
        seq_len = tokens.shape[1]
        x = self.token_embedding(tokens) + self.positional[:, :seq_len]
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm_out(x)

        # Apply masked mean pooling to prevent padded positions from affecting the final classification.
        weights = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def count_parameters(model) -> int:
    """Counts the total number of trainable parameters in the model, correctly handling shared modules."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)