"""Why does my linear arm win on accuracy? I'm comparing what each arm's encoder actually learned.

The result I need to explain: at a matched budget and inside an identical frame, my 
linear arm reaches 0.4425 on CIFAR and 0.3788 on ListOps while the exact-attention arms 
sit around 0.30 and 0.31. When I look at the published LRA reference for a plain Transformer 
(42.44 on Image, 36.37 on ListOps), the linear arm is hitting the target and the 
softmax arms are the ones falling short. My precision probe ruled out bf16 as the cause: 
vanilla scored 0.3804 in fp32 against 0.3882 in bf16, so full precision doesn't rescue it.

That leaves a question about the mechanisms themselves, and I have a specific hypothesis 
to test rather than just speculating. On Pathfinder, every arm collapsed into a 
constant-prediction solution, and my collapse probe located it exactly: `norm_out`'s 
LayerNorm scale was driven from its initial 1.0 toward 0, starving the whole encoder 
of gradients since gamma multiplies everything flowing back. If the softmax arms are 
PARTIALLY doing this on the learnable tasks while the linear arm isn't, it would explain 
both the shortfall against the reference and the design sensitivity I'm already seeing 
in the grid (flash spans 0.216-0.394 across CIFAR designs, while linear never drops below 0.378).

Here is what I'm measuring for every arm at one (task, design), using checkpoints already on disk:

  * norm_out gamma: Initialized at 1.0. I want to see how far it's been driven toward zero, 
    and what fraction of its dimensions are effectively switched off. This is my collapse indicator.
  * representation spread: How much the token representations and pooled vector vary 
    from one input to the next. If it barely changes with the input, it can't support classification.
  * effective rank: The participation ratio of the representation covariance spectrum. 
    A collapsed encoder concentrates its variance into very few directions. This is the one 
    measure here that is mechanism-agnostic and totally comparable across all arms.

What I'm deliberately NOT measuring: attention entropy. It's the obvious choice, but 
it's not comparable across these arms -- Flash never materializes an attention 
matrix, and linear has no softmax distribution to take the entropy of. Comparing 
apples to oranges would just give me a table that looks like evidence but isn't.

Run on a GPU node; it trains nothing and takes a few minutes.

    python cluster/representation_probe.py --task cifar --design-id 0
    python cluster/representation_probe.py --task listops --design-id 5 --seed 0
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from attn_rct.data import build_dataloaders                  # noqa: E402
from attn_rct.model import TransformerClassifier             # noqa: E402
from attn_rct.train import load_manifest_row                 # noqa: E402

VARIANTS = ["vanilla", "flash", "linformer", "linear", "sparse"]


def find_row(manifest, task, design_id, variant, seed):
    """Locates the manifest row for one cell so the geometry comes straight from the source.

    Rebuilding a config by hand is an easy way to accidentally measure a different model 
    from the one I actually trained. The manifest is my record of what ran, so it's the source of truth.

    Returns:
        (row_index, cfg) with cfg already type-cast, or (None, None) if absent.
    """
    import csv
    with open(manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    want = f"{task}_d{design_id:03d}_{variant}_s{seed}"
    for i, row in enumerate(rows, start=1):
        if row["run_id"] == want:
            return i, load_manifest_row(manifest, i)
    return None, None


def locate_checkpoint(root, run_id):
    """Finds a cell's checkpoint, handling the different directory layouts I'm using."""
    root = pathlib.Path(root)
    direct = root / run_id / "last.pt"
    if direct.exists():
        return direct
    hits = sorted(root.glob(f"**/{run_id}/last.pt")) or sorted(
        root.glob(f"**/*{run_id}*/last.pt")
    )
    return hits[0] if hits else None


def encode(model, tokens, mask):
    """Runs the frame's forward pass but stops right before pooling.

    Returns:
        Post-`norm_out` token representations, (B, S, D) -- what the pooling step consumes.
    """
    seq_len = tokens.shape[1]
    x = model.token_embedding(tokens) + model.positional[:, :seq_len]
    for block in model.blocks:
        x = block(x, mask)
    return model.norm_out(x)


def effective_rank(hidden):
    """Calculates the participation ratio of the representation covariance spectrum.

    (sum of eigenvalues)^2 / (sum of squared eigenvalues). This equals D when variance is spread
    evenly across all D directions, and approaches 1 when a single direction takes over. 
    I can read this as "how many dimensions is this encoder actually using". Unlike attention entropy,
    it's defined identically for every arm, which makes it usable here.

    Args:
        hidden: (B, S, D) token representations.

    Returns:
        The participation ratio, as a float.
    """
    flat = hidden.reshape(-1, hidden.shape[-1]).float()
    flat = flat - flat.mean(dim=0, keepdim=True)
    # Eigenvalues of the covariance, via singular values of the centred matrix.
    singular = torch.linalg.svdvals(flat)
    eigen = singular.pow(2)
    total = eigen.sum()
    if total <= 0:
        return 0.0
    return float(total.pow(2) / eigen.pow(2).sum())


def probe_variant(task, design_id, variant, seed, manifest, root, loader, meta, device):
    row_index, cfg = find_row(manifest, task, design_id, variant, seed)
    if cfg is None:
        print(f"  {variant:<10s} not in manifest, skipped")
        return None

    checkpoint = locate_checkpoint(root, cfg.run_id)
    if checkpoint is None:
        print(f"  {variant:<10s} no checkpoint found for {cfg.run_id}, skipped")
        return None

    cfg.attention = cfg.variant
    # I only want the flash arm checking this. A silent fallback here would mean I'm probing 
    # a different kernel than the one that produced the result I'm trying to explain.
    cfg.require_flash = (variant == "flash")

    model = TransformerClassifier(cfg, vocab_size=meta["vocab_size"],
                                  n_classes=meta["n_classes"])
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    tokens, mask, targets = (t.to(device) for t in next(iter(loader)))
    with torch.no_grad():
        hidden = encode(model, tokens, mask)
        logits = model(tokens, mask)

    weights = mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    gamma = model.norm_out.weight.detach().float()
    hidden_f = hidden.float()
    accuracy = (logits.argmax(-1) == targets).float().mean().item()

    return {
        "variant": variant,
        "run_id": cfg.run_id,
        "epoch": state.get("epoch"),
        "best_acc": state.get("best_accuracy"),
        "batch_acc": accuracy,
        "gamma_absmean": float(gamma.abs().mean()),
        "gamma_max": float(gamma.abs().max()),
        "gamma_dead_frac": float((gamma.abs() < 0.1).float().mean()),
        "tok_across": float(hidden_f.std(dim=0).mean()),
        "tok_within": float(hidden_f.std(dim=1).mean()),
        "pooled_spread": float(pooled.float().std(dim=0).mean()),
        "logit_spread": float(logits.float().std(dim=0).mean()),
        "eff_rank": effective_rank(hidden_f),
        "d_model": int(cfg.d_model),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="cifar")
    parser.add_argument("--design-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", default="cluster/runs.csv")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--data-dir", default=None,
                        help="defaults to the manifest's data_dir for this task")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Any arm's row carries the task's data_dir and max_len, so the first one I find 
    # settles the loader configuration for all of them.
    _row, reference = find_row(args.manifest, args.task, args.design_id, "vanilla", args.seed)
    if reference is None:
        raise SystemExit(f"no manifest row for {args.task} d{args.design_id:03d}")
    data_dir = args.data_dir or reference.data_dir

    _train, val_loader, meta = build_dataloaders(
        args.task, data_dir, int(reference.max_len),
        batch_size=args.batch, seed=0, num_workers=0,
    )

    print("=" * 108)
    print(f"Representation probe | {args.task} d{args.design_id:03d} seed {args.seed} | "
          f"d_model {reference.d_model} depth {reference.depth} lr {reference.lr} | "
          f"device {device}")
    print(f"meta: {meta}")
    print("=" * 108)

    rows = []
    for variant in VARIANTS:
        try:
            result = probe_variant(args.task, args.design_id, variant, args.seed,
                                   args.manifest, args.checkpoint_root,
                                   val_loader, meta, device)
        except Exception as err:                       # noqa: BLE001
            print(f"  {variant:<10s} FAILED ({type(err).__name__}: {err})")
            result = None
        if result:
            rows.append(result)

    if not rows:
        raise SystemExit("no arm could be probed; check --checkpoint-root")

    header = (f"{'arm':<10s} {'best':>6s} {'gamma|mean|':>11s} {'gamma off%':>10s} "
              f"{'eff rank':>9s} {'/d_model':>9s} {'tok across':>11s} "
              f"{'pooled':>10s} {'logits':>9s}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['variant']:<10s} {r['best_acc']:>6.4f} {r['gamma_absmean']:>11.4f} "
              f"{100 * r['gamma_dead_frac']:>9.1f}% {r['eff_rank']:>9.1f} "
              f"{r['eff_rank'] / r['d_model']:>9.3f} {r['tok_across']:>11.3e} "
              f"{r['pooled_spread']:>10.3e} {r['logit_spread']:>9.3e}")

    print("\n" + "=" * 108)
    print("How to read this")
    print("=" * 108)
    print("  gamma|mean| starts at 1.0 for every arm since norm_out is initialized that")
    print("  way and all arms share it. If an arm's gamma has been driven far below 1,")
    print("  it's been pushed toward a constant-prediction solution. Since gamma multiplies")
    print("  the gradient into the whole encoder, this becomes a self-reinforcing trap.")
    print()
    print("  If the softmax arms show low gamma / low effective rank and the linear arm doesn't,")
    print("  my accuracy result is actually about training stability under this frame and budget,")
    print("  not representational capacity. The design sensitivity in the grid (flash 0.216-0.394,")
    print("  linear never below 0.378) is just the same phenomenon seen from the outside.")
    print()
    print("  If every arm looks healthy and only accuracy differs, then the effect genuinely")
    print("  comes down to what each mechanism can represent, and the collapse story doesn't")
    print("  apply to the learnable tasks at all.")


if __name__ == "__main__":
    main()