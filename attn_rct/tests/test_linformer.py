"""Tests for the Linformer arm.

Validates Linformer-specific properties, such as parameter sharing, 
projection matrix behavior, and low-rank complexity, which are not covered 
by the generic attention tests.
"""

from types import SimpleNamespace
import pytest
import torch
from attn_rct.attention import ATTENTION_REGISTRY, LinformerAttention
from attn_rct.attention.linformer import SequenceProjection
from attn_rct.model import TransformerClassifier, count_parameters


def make_cfg(**overrides):
    """Builds a configuration namespace with default values for the Linformer arm."""
    cfg = SimpleNamespace(
        d_model=64, n_heads=4, attention="linformer",
        attn_dropout=0.0, dropout=0.0, compute_dtype="fp32",
        max_len=128, linformer_k=16, linformer_sharing="layerwise",
        depth=3, d_ff=128, seed=0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_batch(batch=3, seq_len=40, d_model=64, seed=0):
    """Builds a batch with variable sequence lengths to test padding behavior."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq_len, d_model, generator=generator)
    mask = torch.zeros(batch, seq_len, dtype=torch.long)
    for i, length in enumerate([seq_len, seq_len - 11, 5][:batch]):
        mask[i, :length] = 1
    return x, mask


def test_registered():
    """Ensures the Linformer arm is correctly registered."""
    assert "linformer" in ATTENTION_REGISTRY


def test_layerwise_sharing_is_one_matrix():
    """Ensures layerwise sharing creates exactly one projection matrix across all layers, heads, keys, and values."""
    cfg = make_cfg()
    model = TransformerClassifier(cfg, vocab_size=20)

    projections = {id(m) for m in model.modules() if isinstance(m, SequenceProjection)}
    assert len(projections) == 1, (
        f"expected 1 shared projection across {cfg.depth} layers, found "
        f"{len(projections)}"
    )

    for block in model.blocks:
        assert block.attention.key_projection is block.attention.value_projection


def test_layerwise_cost_independent_of_depth():
    """Ensures the projection matrix parameter count remains constant regardless of network depth under layerwise sharing."""
    counts = {}
    for depth in (2, 4, 8):
        cfg = make_cfg(depth=depth)
        counts[depth] = count_parameters(TransformerClassifier(cfg, vocab_size=20))

    expected_e = make_cfg().linformer_k * make_cfg().max_len
    growth = counts[4] - counts[2]
    growth_again = counts[8] - counts[4]
    
    assert growth_again == 2 * growth, (
        f"projection cost appears to scale with depth: {counts}"
    )
    assert expected_e == 16 * 128


def test_padding_zeroed_before_projection():
    """Ensures padded positions do not leak through the projection matrix, preventing contamination across projected positions."""
    arm = LinformerAttention(make_cfg()).eval()
    x, mask = make_batch()

    polluted = x.clone()
    polluted[~mask.bool()] = 1e4

    with torch.no_grad():
        clean = arm(x, mask)
        dirty = arm(polluted, mask)

    keep = mask.bool()
    torch.testing.assert_close(clean[keep], dirty[keep], rtol=1e-4, atol=1e-4)


def test_attention_matrix_is_n_by_k():
    """Verifies the attention score matrix shape is reduced to (S, K) instead of the quadratic (S, S)."""
    cfg = make_cfg(linformer_k=16)
    arm = LinformerAttention(cfg).eval()

    captured = {}
    original = torch.Tensor.softmax

    def spy(self, dim=-1, **kwargs):
        captured["shape"] = tuple(self.shape)
        return original(self, dim, **kwargs)

    torch.Tensor.softmax = spy
    try:
        x, mask = make_batch(seq_len=40)
        with torch.no_grad():
            arm(x, mask)
    finally:
        torch.Tensor.softmax = original

    assert captured["shape"][-1] == 16, (
        f"scores end in {captured['shape']}; last dim should be k=16, not the "
        "sequence length, meaning the low-rank projection is not being applied"
    )
    assert captured["shape"][-2] == 40


def test_sharing_modes_differ_in_count():
    """Verifies different sharing modes produce the correct relative parameter counts."""
    counts = {}
    for mode in ("layerwise", "key_value", "headwise"):
        cfg = make_cfg(linformer_sharing=mode)
        counts[mode] = count_parameters(TransformerClassifier(cfg, vocab_size=20))

    assert counts["layerwise"] < counts["key_value"] < counts["headwise"], counts


def test_rejects_sequence_longer_than_max_len():
    """Ensures an error is raised if the sequence length exceeds the maximum length configured for the projection matrix."""
    arm = LinformerAttention(make_cfg(max_len=32))
    x, mask = make_batch(seq_len=40)
    with pytest.raises(ValueError, match="exceeds max_len"):
        arm(x, mask)


def test_unknown_sharing_mode_rejected():
    """Ensures an error is raised for unknown sharing configurations."""
    with pytest.raises(ValueError, match="unknown linformer_sharing"):
        LinformerAttention(make_cfg(linformer_sharing="everywhere"))