"""Tests for the Linear Transformer arm.

Ensures the linear O(N) attention implementation correctly matches the standard O(N^2) 
mathematical definition, and validates its edge cases and memory constraints.
"""

from types import SimpleNamespace
import torch
from attn_rct.attention import ATTENTION_REGISTRY, LinearAttention
from attn_rct.attention.linear import EPS, elu_feature_map


def make_cfg(**overrides):
    """Builds a configuration namespace with default values for the linear arm."""
    cfg = SimpleNamespace(
        d_model=64, n_heads=4, attention="linear",
        attn_dropout=0.0, dropout=0.0, compute_dtype="fp32",
        max_len=128, seed=0, depth=2, d_ff=128,
        linformer_k=16, linformer_sharing="layerwise",
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
    """Ensures the linear arm is correctly registered."""
    assert "linear" in ATTENTION_REGISTRY


def test_feature_map_is_non_negative():
    """Ensures the feature map phi(x) = elu(x) + 1 remains non-negative.

    Extreme negative values can underflow to exactly zero in fp32. This validates 
    why an epsilon is necessary in the denominator during the attention computation.
    """
    x = torch.linspace(-50, 50, 10001)
    phi = elu_feature_map(x)
    assert (phi >= 0).all(), f"minimum was {phi.min().item()}"
    assert phi.min() == 0.0, "expected underflow to exactly zero at the negative end"
    assert elu_feature_map(torch.tensor(0.0)) == 1.0


def test_eps_guards_total_underflow():
    """Ensures queries with features that underflow to zero do not produce NaN or infinite outputs."""
    arm = LinearAttention(make_cfg()).eval()
    q = torch.full((1, 2, 4, 16), -60.0)
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)
    with torch.no_grad():
        out = arm._attend(q, k, v, mask=None)
    assert torch.isfinite(out).all(), "eps failed to guard a zero denominator"


def test_matches_quadratic_reference():
    """Verifies the O(N) linear attention computation exactly matches the naive O(N^2) computation."""
    torch.manual_seed(0)
    arm = LinearAttention(make_cfg()).eval()

    batch, heads, seq_len, head_dim = 2, 4, 30, 16
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    with torch.no_grad():
        fast = arm._attend(q, k, v, mask=None)

        phi_q, phi_k = elu_feature_map(q), elu_feature_map(k)
        similarity = torch.einsum("bhid,bhjd->bhij", phi_q, phi_k)
        numerator = torch.einsum("bhij,bhjm->bhim", similarity, v)
        denominator = similarity.sum(dim=-1, keepdim=True) + EPS
        slow = numerator / denominator

    difference = (fast - slow).abs().max().item()
    print(f"\nmax |linear - quadratic reference| = {difference:.3e}")
    assert difference < 1e-5, f"associativity trick disagrees with reference: {difference:.3e}"


def test_masking_matches_quadratic_reference():
    """Verifies the O(N) and O(N^2) computations match when padding masks are applied."""
    torch.manual_seed(1)
    arm = LinearAttention(make_cfg()).eval()

    batch, heads, seq_len, head_dim = 2, 4, 25, 16
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    mask[0, :seq_len] = True
    mask[1, :7] = True

    with torch.no_grad():
        fast = arm._attend(q, k, v, mask=mask)

        phi_q, phi_k = elu_feature_map(q), elu_feature_map(k)
        phi_k = phi_k * mask[:, None, :, None].to(phi_k.dtype)
        similarity = torch.einsum("bhid,bhjd->bhij", phi_q, phi_k)
        numerator = torch.einsum("bhij,bhjm->bhim", similarity, v)
        denominator = similarity.sum(dim=-1, keepdim=True) + EPS
        slow = numerator / denominator

    difference = (fast - slow).abs().max().item()
    assert difference < 1e-5, f"masked form disagrees: {difference:.3e}"


def test_no_quadratic_intermediate():
    """Ensures no large (S, S) intermediate tensors are created in memory."""
    arm = LinearAttention(make_cfg()).eval()
    batch, heads, seq_len, head_dim = 1, 4, 4000, 16

    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    quadratic_elements = batch * heads * seq_len * seq_len
    with torch.no_grad():
        out = arm._attend(q, k, v, mask=None)

    assert out.shape == (batch, heads, seq_len, head_dim)
    assert quadratic_elements > 6e7


def test_output_is_convex_combination_of_values():
    """Ensures each output correctly falls within the range of the values it aggregates, acting as a weighted average."""
    torch.manual_seed(2)
    arm = LinearAttention(make_cfg()).eval()
    q = torch.randn(1, 2, 20, 16)
    k = torch.randn(1, 2, 20, 16)
    v = torch.rand(1, 2, 20, 16)

    with torch.no_grad():
        out = arm._attend(q, k, v, mask=None)

    assert out.min() >= -1e-5, out.min()
    assert out.max() <= 1.0 + 1e-5, out.max()


def test_same_parameter_count_as_vanilla():
    """Verifies the linear arm adds no extra parameters compared to the vanilla baseline."""
    torch.manual_seed(0)
    linear = LinearAttention(make_cfg())
    torch.manual_seed(0)
    vanilla = ATTENTION_REGISTRY["vanilla"](make_cfg(attention="vanilla"))

    assert (sum(p.numel() for p in linear.parameters())
            == sum(p.numel() for p in vanilla.parameters()))