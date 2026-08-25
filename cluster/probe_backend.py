"""Verify that FA-2 actually engages on this node, and that flash matches vanilla.

Capability is not execution: an sm_86 device can still run a different kernel, for
instance when a boolean attn_mask is passed to SDPA and disqualifies the flash backend.
The pilot's probe read capability and inferred the rest, which would have logged
mem-efficient runs as FA-2.

This is a gate, not a formality. If it does not print VERIFIED, no efficiency number
from the baseline arm is trustworthy.
"""

import os
import pathlib
import socket
import sys
from types import SimpleNamespace

import torch

# We resolve the repo root from this file rather than hardcoding it, so the probe works
# wherever the project is checked out.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from attn_rct.attention import FlashAttention, VanillaAttention
except ModuleNotFoundError as err:
    print(f"FATAL: cannot import the attention package ({err}).")
    print(f"  looked in: {REPO_ROOT}")
    print("  expected:  <repo>/attn_rct/attention/{base,vanilla,flash}.py")
    print("  -> copy the package from your dev machine; the cluster copy is incomplete.")
    sys.exit(1)


def cfg(**overrides):
    """Build a probe config.

    Args:
        **overrides: fields to replace.

    Returns:
        The config namespace.
    """
    base = SimpleNamespace(
        d_model=256, n_heads=8, attention="flash",
        attn_dropout=0.0, compute_dtype="bf16", require_flash=False,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def check_exactness(device):
    """Compare the flash arm against the vanilla reference on a ragged batch.

    Both arms run in the SAME dtype: this is a code-correctness check, not the
    experiment. Comparing bf16 against fp32 would show ~1e-2 disagreement and prove
    nothing about whether the masking is right.

    Args:
        device: CUDA device.

    Returns:
        (ok, path_taken, difference) where difference is over real tokens only.
    """
    torch.manual_seed(1903697)
    vanilla = VanillaAttention(cfg(attention="vanilla")).to(device).eval()
    torch.manual_seed(1903697)
    flash = FlashAttention(cfg(require_flash=True)).to(device).eval()

    batch, seq_len = 4, 2000
    x = torch.randn(batch, seq_len, 256, device=device)
    mask = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    for i, length in enumerate([2000, 1500, 900, 17]):   # includes a very short one
        mask[i, :length] = 1

    # We compare only at real token positions. The varlen path drops padded queries, so
    # pad_input restores them as zeros, while vanilla computes an ordinary output
    # there. Both are discarded downstream, so including them would measure "zero"
    # against "undefined".
    keep = mask.bool()
    with torch.no_grad():
        out_vanilla = vanilla(x, mask)
        out_flash = flash(x, mask)
        difference = (out_vanilla[keep] - out_flash[keep]).abs().max().item()
        difference_all = (out_vanilla - out_flash).abs().max().item()

    print("path taken:", flash._path_logged)
    print(f"max |vanilla - flash| over padding too = {difference_all:.3e} (expected large)")
    print(f"max |vanilla - flash| = {difference:.3e}")

    ok = (flash._path_logged == "flash_varlen_fa2") and (difference < 5e-2)
    return ok, flash._path_logged, difference


def check_wandb_reachable():
    """Report whether this node can reach W&B, so stage 5 does not find out the hard way."""
    if not os.environ.get("WANDB_API_KEY"):
        print("wandb: WANDB_API_KEY not set in this job's environment")
        return

    try:
        socket.create_connection(("api.wandb.ai", 443), timeout=10).close()
        print("wandb: compute node has outbound network -- online mode fine")
    except Exception as err:  # noqa: BLE001
        print(f"wandb: no outbound network ({err!r})")
        print("  -> set WANDB_MODE=offline in the job scripts and `wandb sync` later")


def main():
    print("=" * 70)
    print("node:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("torch:", torch.__version__)

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible.")
        print("  -> if sinfo showed gpu GRES, the job needs #SBATCH --gres=gpu:1")
        sys.exit(1)

    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    print(f"capability: sm_{major}{minor}")

    if major < 8:
        print(f"FAIL: FA-2 needs sm_80+. sm_{major}{minor} cannot run it.")
        print("  -> use partition bigbatch (RTX 3090, sm_86).")
        print("  -> biggpu (RTX 8000) is Turing sm_75 and also cannot.")
        sys.exit(1)

    print("backend:", FlashAttention.measure_backend(device))

    ok, _, _ = check_exactness(device)
    print("VERIFIED" if ok else "FAIL: did not take the FA-2 path, or exactness violated")

    check_wandb_reachable()

    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
