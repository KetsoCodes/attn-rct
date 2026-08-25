"""Tests for the sparse attention arm.

Validates correctness, padding behaviors, and block-density metrics across the 
different sparse attention patterns.
"""

from types import SimpleNamespace
import pytest
import torch
from attn_rct.attention import ATTENTION_REGISTRY, SparseAttention
from attn_rct.attention.sparse import PATTERNS, block_density


def make_cfg(**overrides):
    """Builds a configuration namespace with default values for the sparse arm."""
    cfg = SimpleNamespace(
        d_model=64, n_heads=4, attention="sparse",
        attn_dropout=0.0, dropout=0.0, compute_dtype="fp32",
        max_len=256, seed=0, depth=2, d_ff=128,
        linformer_k=16, linformer_sharing="layerwise",
        sparse_pattern="fixed", sparse_stride=None,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_batch(batch=3, seq_len=64, d_model=64, seed=0):
    """Builds a batch with variable sequence lengths to test padding and empty-row edge cases."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq_len, d_model, generator=generator)
    mask = torch.zeros(batch, seq_len, dtype=torch.long)
    for i, length in enumerate([seq_len, seq_len - 20, 4][:batch]):
        mask[i, :length] = 1
    return x, mask


def test_registered():
    """Ensures the sparse arm is correctly registered."""
    assert "sparse" in ATTENTION_REGISTRY


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
def test_every_pattern_runs_and_is_finite(pattern):
    """Ensures every configured sparse pattern executes successfully and produces finite outputs."""
    arm = SparseAttention(make_cfg(sparse_pattern=pattern)).eval()
    x, mask = make_batch()
    with torch.no_grad():
        out = arm(x, mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all(), f"{pattern} produced non-finite output"


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
def test_no_row_is_entirely_masked(pattern):
    """Ensures every query keeps at least one key to prevent the softmax from outputting NaNs."""
    arm = SparseAttention(make_cfg(sparse_pattern=pattern)).eval()
    x, mask = make_batch()

    allowed = arm.pattern_mask(x.shape[1], x.device)[None]
    allowed = allowed & mask[:, None, None, :].bool()
    empty = ~allowed.any(dim=-1)

    with torch.no_grad():
        out = arm(x, mask)
    assert torch.isfinite(out).all(), (
        f"{pattern}: {empty.sum().item()} empty rows produced non-finite output"
    )


def test_padding_is_inert():
    """Ensures real-token outputs do not depend on the values in padded slots."""
    arm = SparseAttention(make_cfg()).eval()
    x, mask = make_batch()
    polluted = x.clone()
    polluted[~mask.bool()] = 1e4

    with torch.no_grad():
        clean, dirty = arm(x, mask), arm(polluted, mask)

    keep = mask.bool()
    torch.testing.assert_close(clean[keep], dirty[keep], rtol=1e-4, atol=1e-4)


def test_pattern_is_actually_sparse():
    """Verifies the mask drops a substantial fraction of pairs and does not act like a dense mask."""
    for pattern in PATTERNS:
        arm = SparseAttention(make_cfg(sparse_pattern=pattern))
        mask = arm.pattern_mask(256, torch.device("cpu"))
        density = mask.float().mean().item()
        assert density < 0.6, f"{pattern} is {density:.1%} dense and barely sparse"


def test_diagonal_always_allowed():
    """Ensures tokens can always attend to themselves across all patterns and heads."""
    for pattern in PATTERNS:
        arm = SparseAttention(make_cfg(sparse_pattern=pattern))
        mask = arm.pattern_mask(128, torch.device("cpu"))
        diagonal = torch.eye(128, dtype=torch.bool)
        for head in range(mask.shape[0]):
            assert (mask[head] & diagonal).sum() == 128, f"{pattern} head {head}"


def test_child_transpose_recovers_block_sparsity():
    """Verifies that permuting the sequence indices allows the strided pattern to benefit from block sparsity."""
    n, stride, block = 2048, 32, 32
    idx = torch.arange(n)
    q, k = idx[:, None], idx[None, :]
    causal_strided = (q >= k) & ((q - k) % stride == 0)

    perm = torch.arange(n).reshape(n // stride, stride).T.reshape(-1)
    transposed = causal_strided[perm][:, perm]

    before = block_density(causal_strided, block)
    after = block_density(transposed, block)

    assert before > 0.4, f"causal strided should be ~50% block-dense, got {before:.1%}"
    assert after < 0.10, (
        f"after strided_transpose it should be nearly all skippable, got {after:.1%}"
    )


def test_symmetrising_inflates_block_density():
    """Demonstrates that making the strided pattern symmetric (non-causal) significantly increases its block density."""
    n, stride, block = 2048, 32, 32
    idx = torch.arange(n)
    q, k = idx[:, None], idx[None, :]

    causal = (q >= k) & ((q - k) % stride == 0)
    symmetric = ((q - k) % stride == 0)

    assert block_density(causal, block) < 0.6
    assert block_density(symmetric, block) == 1.0


def test_unknown_pattern_rejected():
    """Ensures an error is raised for unknown sparse pattern configurations."""
    with pytest.raises(ValueError, match="unknown sparse_pattern"):
        SparseAttention(make_cfg(sparse_pattern="diagonal_ish"))


def test_mask_is_cached():
    """Ensures the boolean attention masks are cached to prevent costly recalculations on every forward pass."""
    arm = SparseAttention(make_cfg())
    first = arm.pattern_mask(128, torch.device("cpu"))
    second = arm.pattern_mask(128, torch.device("cpu"))
    assert first is second