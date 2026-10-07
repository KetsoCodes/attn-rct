"""Why my Pathfinder runs are stuck at chance: measuring the exact state they're trapped in.

The recipe probe proved that the training budget wasn't the issue—in fact, the way it failed is the actual finding. At a learning rate of 1e-2, the loss didn't just stay high; it completely FROZE. In my `pfrecipe_lra_faithful` run, the validation loss sat at exactly 0.6931468685150146 for twelve straight epochs, with the training loss pinned perfectly at ln(2). In `pfrecipe_lra_lr_ourdim`, it froze at 0.7956, which is even worse: confidently wrong. 

When loss is bit-identical across epochs, the model isn't learning slowly. It means the network's output is completely ignoring the input, and the gradients are gone. The 1e-3 run is slightly different—the loss still wiggles in the fourth decimal place, so it's alive, but it's still not learning. 

So, I have two questions left that I can answer just by probing the saved checkpoints, without wasting GPU hours on more training:

1. WHERE is it losing the input? The encoder might actually be working (producing input-dependent representations), but then my mean-pooling step just averages the signal away. Or, the encoder itself might be dead. I can figure this out easily by measuring the variance across examples before and after pooling.

2. Is this a true fixed point? If the gradient norm is practically zero, running more epochs is completely useless and I need to kill the jobs.

The readout comparison in check 3 is the most important one for my thesis write-up. It tests three different pooling methods on the EXACT SAME frozen encoder to isolate the readout layer. Right now, my frame uses mean-pooling over all 1024 positions. But 62% of those positions are just the black background, which `padding_idx=0` forces to a zero vector. So, I'm basically taking a tiny signal and dividing it by a massive constant. If max-pooling or bright-pixel-only pooling suddenly brings back the variance, then I know my readout layer is the bottleneck—and that's a precise, reportable mechanism. If none of them work, the encoder has just collapsed.

Run on a GPU node; it trains nothing.

    python cluster/pathfinder_collapse_probe.py
    python cluster/pathfinder_collapse_probe.py --checkpoint-dir checkpoints/pfrecipe_lra_faithful
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from attn_rct.data import build_dataloaders          # noqa: E402
from attn_rct.model import TransformerClassifier     # noqa: E402

DEFAULT_DATA = "/datasets/fmnisi/pathfinder"

# The three recipe cells, with the geometry each was run at. The checkpoint does not record
# its own d_model, so it has to be rebuilt at the right shape to load.
CELLS = {
    "pfrecipe_lra_faithful":  {"d_model": 128, "depth": 4, "lr": 1e-2},
    "pfrecipe_lra_lr_ourdim": {"d_model": 256, "depth": 4, "lr": 1e-2},
    "pfrecipe_ourlr_long":    {"d_model": 128, "depth": 4, "lr": 1e-3},
}


def make_cfg(d_model, depth, max_len):
    """The frame's configuration, with dropout off so measurements are deterministic."""
    return SimpleNamespace(
        d_model=d_model, n_heads=8, depth=depth, d_ff=4 * d_model,
        dropout=0.0, attn_dropout=0.0,
        max_len=max_len, attention="flash", variant="flash", seed=0,
        linformer_k=256, linformer_sharing="layerwise",
        sparse_pattern="fixed", sparse_stride=45,
        compute_dtype="bf16", require_flash=False,
    )


def encode(model, tokens, mask):
    """Run the frame's forward pass but stop before pooling.

    Returns:
        The post-`norm_out` token representations, shape (B, S, D) -- exactly the tensor
        the model's own pooling step consumes.
    """
    seq_len = tokens.shape[1]
    x = model.token_embedding(tokens) + model.positional[:, :seq_len]
    for block in model.blocks:
        x = block(x, mask)
    return model.norm_out(x)


def readouts(hidden, tokens, mask):
    """Pool the same token representations three different ways.

    Args:
        hidden: (B, S, D) token representations from `encode`.
        tokens: (B, S) token ids, used to find the bright (non-background) positions.
        mask: (B, S) the frame's padding mask.

    Returns:
        A dict of name -> (B, D) pooled representation.
    """
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    frame_mean = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    # Background is token 0, which padding_idx pins to the zero vector. Pooling over only
    # the informative positions removes the constant that dilutes the frame's mean.
    bright = (tokens > 0).unsqueeze(-1).to(hidden.dtype)
    bright_mean = (hidden * bright).sum(1) / bright.sum(1).clamp(min=1.0)

    return {
        "mean over all 1024 (the frame)": frame_mean,
        "mean over bright positions only": bright_mean,
        "max over positions": hidden.max(dim=1).values,
        "first position (CLS-style)": hidden[:, 0],
    }


def across_example_signal(pooled):
    """How much does this representation vary from one input to the next?

    A readout whose output is the same vector for every input carries no information about
    the input, whatever the encoder computed. This is that quantity in one number: the
    standard deviation across examples, averaged over dimensions, normalised by the overall
    scale so the three readouts are comparable despite different magnitudes.
    """
    pooled = pooled.float()
    spread = pooled.std(dim=0).mean().item()
    scale = pooled.abs().mean().item() + 1e-12
    return spread, spread / scale


def report_cell(name, geometry, checkpoint_dir, loader, meta, device, batch):
    path = pathlib.Path(checkpoint_dir) / "last.pt"
    if not path.exists():
        print(f"  {name}: no checkpoint at {path}, skipped")
        return

    state = torch.load(path, map_location=device)
    cfg = make_cfg(geometry["d_model"], geometry["depth"], meta["max_len"])
    model = TransformerClassifier(cfg, vocab_size=meta["vocab_size"],
                                  n_classes=meta["n_classes"])
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    tokens, mask, targets = (t.to(device) for t in batch)

    print(f"\n{'=' * 74}")
    print(f"{name}   (d_model {geometry['d_model']}, lr {geometry['lr']:g}, "
          f"epoch {state['epoch']}, best acc {state['best_accuracy']:.4f})")
    print("=" * 74)

    with torch.no_grad():
        hidden = encode(model, tokens, mask)
        logits = model(tokens, mask)

    # 1. Is the encoder's output input-dependent at all, before any pooling?
    per_token = hidden.float()
    token_spread = per_token.std(dim=0).mean().item()
    within_seq_spread = per_token.std(dim=1).mean().item()
    print(f"  token reps, spread ACROSS examples : {token_spread:.3e}")
    print(f"  token reps, spread WITHIN a sequence: {within_seq_spread:.3e}")
    if within_seq_spread < 1e-4:
        print("    -> every position holds the same vector: token-uniformity collapse.")

    # 2. Which readout, if any, preserves that input dependence?
    print("  readout                            abs spread   relative")
    for label, pooled in readouts(hidden, tokens, mask).items():
        spread, relative = across_example_signal(pooled)
        print(f"    {label:<33s} {spread:.3e}   {relative:.2e}")

    # 3. Are the logits constant, and is the loss at ln(k)?
    logit_spread = logits.float().std(dim=0).mean().item()
    mean_logits = logits.float().mean(dim=0)
    loss = nn.CrossEntropyLoss()(logits, targets).item()
    import math
    print(f"  logits: spread across examples {logit_spread:.3e}, "
          f"mean {[round(v, 4) for v in mean_logits.tolist()]}")
    print(f"  loss {loss:.10f}   ln({meta['n_classes']}) = "
          f"{math.log(meta['n_classes']):.10f}")

    # 4. Is this a fixed point? A vanished gradient means more epochs cannot help.
    model.train()
    model.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss()(model(tokens, mask), targets).backward()
    total = sum(float(p.grad.float().pow(2).sum()) for p in model.parameters()
                if p.grad is not None) ** 0.5
    head_grad = float(model.head.weight.grad.float().norm())
    print(f"  grad norm: total {total:.3e}, head {head_grad:.3e}")
    if total < 1e-6:
        print("    -> gradient has vanished. This is a FIXED POINT: further epochs at this")
        print("       learning rate cannot move it, and the chained rounds are wasted GPU.")

    # 5. Has the frame's own scale collapsed?
    gamma = model.norm_out.weight.float()
    print(f"  norm_out gamma: mean {gamma.mean():.4f}, min {gamma.min():.4f}, "
          f"max {gamma.max():.4f}")
    print(f"  head weight norm {float(model.head.weight.float().norm()):.4f}, "
          f"bias {[round(v, 4) for v in model.head.bias.float().tolist()]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--task", default="pathfinder")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--checkpoint-root", default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pathfinder collapse probe | device={device}")

    _train, val_loader, meta = build_dataloaders(
        args.task, args.data_dir, args.max_len,
        batch_size=args.batch, seed=0, num_workers=0,
    )
    batch = next(iter(val_loader))
    print(f"meta: {meta}")

    root = pathlib.Path(args.checkpoint_root)
    for name, geometry in CELLS.items():
        try:
            report_cell(name, geometry, root / name, val_loader, meta, device, batch)
        except Exception as err:                       # noqa: BLE001
            print(f"\n  {name}: FAILED ({type(err).__name__}: {err})")

    print(f"\n{'=' * 74}")
    print("How to read this")
    print("=" * 74)
    print("  within-sequence spread ~0   -> token-uniformity collapse in the encoder; the")
    print("     high learning rate destroyed the representation and the readout is innocent.")
    print("  across-example spread large before pooling but ~0 after the frame's mean, while")
    print("     bright-only or max pooling keeps it -> the READOUT is the bottleneck. Mean-")
    print("     pooling over 1024 positions, 62% of them a frozen zero vector, divides a thin")
    print("     signal by a large constant. That is a precise, reportable mechanism.")
    print("  everything ~0 including before pooling, with grad norm ~0 -> a dead fixed point;")
    print("     report the null with this evidence and spend no more GPU on it.")


if __name__ == "__main__":
    main()