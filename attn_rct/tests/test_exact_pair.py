"""Tests for the exact-attention pair: the vanilla reference and the FlashAttention-2 baseline.

To run these tests:
    python -m pytest attn_rct/tests/ -v
"""

from types import SimpleNamespace
import pytest
import torch
from attn_rct.attention import ATTENTION_REGISTRY, FlashAttention, VanillaAttention

ARM_NAMES = sorted(ATTENTION_REGISTRY)


def make_cfg(**overrides):
    """Builds a configuration namespace with default values for testing all arms."""
    cfg = SimpleNamespace(
        d_model=64, n_heads=4, attention="vanilla",
        attn_dropout=0.0, compute_dtype="fp32", require_flash=False,
        max_len=64, seed=0,
        linformer_k=16, linformer_sharing="layerwise",
        sparse_pattern="fixed", sparse_stride=None,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_batch(batch=3, seq_len=17, d_model=64, seed=0):
    """Builds a batch with variable sequence lengths to test padding behavior."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq_len, d_model, generator=generator)
    mask = torch.zeros(batch, seq_len, dtype=torch.long)
    for i, length in enumerate([seq_len, seq_len - 5, 3][:batch]):
        mask[i, :length] = 1
    return x, mask


# Interface Tests

@pytest.mark.parametrize("name", ARM_NAMES)
def test_shape_and_grad(name):
    """Ensures every arm outputs the correct shape and is differentiable."""
    arm = ATTENTION_REGISTRY[name](make_cfg())
    x, mask = make_batch()
    x.requires_grad_(True)

    out = arm(x, mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all(), "non-finite output"

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for pname, param in arm.named_parameters():
        assert param.grad is not None, f"{name}.{pname} received no gradient"


@pytest.mark.parametrize("name", ARM_NAMES)
def test_padding_is_inert(name):
    """Ensures real-token outputs do not depend on the values in padded slots."""
    arm = ATTENTION_REGISTRY[name](make_cfg()).eval()
    x, mask = make_batch()

    polluted = x.clone()
    polluted[~mask.bool()] = 1e3

    with torch.no_grad():
        out_clean = arm(x, mask)
        out_polluted = arm(polluted, mask)

    keep = mask.bool()
    torch.testing.assert_close(out_clean[keep], out_polluted[keep], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", ARM_NAMES)
def test_full_mask_equals_no_mask(name):
    """Ensures masked and unmasked calls agree when no padding is present."""
    arm = ATTENTION_REGISTRY[name](make_cfg()).eval()
    x = torch.randn(2, 11, 64, generator=torch.Generator().manual_seed(1))
    ones = torch.ones(2, 11, dtype=torch.long)

    with torch.no_grad():
        torch.testing.assert_close(arm(x, ones), arm(x, None), rtol=1e-5, atol=1e-6)


def test_rejects_all_pad_sequence():
    """Ensures batches with entirely padded sequences raise an error immediately."""
    arm = VanillaAttention(make_cfg())
    x, mask = make_batch()
    mask[1] = 0
    with pytest.raises(ValueError, match="entirely padding"):
        arm(x, mask)


# Pairing Guarantee Tests

def test_arms_share_initialisation():
    """Ensures all arms start from identical shared weights when using the same seed."""
    reference = None
    for name in ARM_NAMES:
        torch.manual_seed(1903697)
        arm = ATTENTION_REGISTRY[name](make_cfg(attention=name))
        weights = {k: v.clone() for k, v in arm.state_dict().items()
                   if k.startswith(("qkv_proj", "out_proj"))}
        assert weights, f"{name} exposes no shared projections"
        
        if reference is None:
            reference = weights
            continue
            
        for key, value in weights.items():
            torch.testing.assert_close(
                value, reference[key], rtol=0, atol=0,
                msg=f"{name}.{key} differs from the reference arm at init",
            )


# Exactness Tests

def test_flash_matches_vanilla():
    """Verifies FlashAttention matches the Vanilla reference implementation exactly."""
    torch.manual_seed(0)
    vanilla = VanillaAttention(make_cfg()).eval()
    torch.manual_seed(0)
    flash = FlashAttention(make_cfg()).eval()

    x, mask = make_batch()
    with torch.no_grad():
        difference = (vanilla(x, mask) - flash(x, mask)).abs().max().item()

    print(f"\nmax |vanilla - flash| = {difference:.3e}  (path={flash._path_logged})")
    assert difference < 1e-5, f"exactness violated: {difference:.3e}"


def fa2_available():
    """Checks if the machine supports FlashAttention-2 (requires sm_80+ GPU and flash_attn)."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 8:
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not fa2_available(),
                    reason="needs sm_80+ and the flash_attn library (run on bigbatch)")
def test_flash_varlen_matches_vanilla_cuda():
    """Validates FlashAttention exactness on a CUDA device using bfloat16."""
    device = torch.device("cuda")
    cfg = make_cfg(compute_dtype="bf16", require_flash=True)

    torch.manual_seed(0)
    vanilla = VanillaAttention(make_cfg(compute_dtype="bf16")).to(device).eval()
    torch.manual_seed(0)
    flash = FlashAttention(cfg).to(device).eval()

    x, mask = make_batch(batch=3, seq_len=2000, seed=7)
    x, mask = x.to(device), mask.to(device)

    # Compare only at real token positions since the padding handling differs 
    # slightly internally, even though both are safely discarded downstream.
    keep = mask.bool()
    with torch.no_grad():
        difference = (vanilla(x, mask)[keep] - flash(x, mask)[keep]).abs().max().item()

    print(f"\nbackend: {FlashAttention.measure_backend(device)}")
    print(f"path taken: {flash._path_logged}")
    print(f"max |vanilla - flash| = {difference:.3e}")

    assert flash._path_logged == "flash_varlen_fa2", "did not take the FA-2 path"
    assert difference < 5e-2, f"exactness violated in bf16: {difference:.3e}"


@pytest.mark.skipif(fa2_available(), reason="only meaningful where FA-2 is unavailable")
def test_require_flash_fails_loudly():
    """Ensures require_flash=True raises an error instead of quietly using a fallback kernel."""
    flash = FlashAttention(make_cfg(require_flash=True))
    x, mask = make_batch()
    with pytest.raises(RuntimeError, match="Refusing to silently measure"):
        flash(x, mask)