"""Generate synthetic results in the exact schema train.py writes.

The analysis pipeline reads results/*.json, but the real grid takes days and we need to
develop and test the statistics before it finishes. This module writes fake result files
that are byte-compatible with what train.py's build_result produces, so the collector and
every downstream stage cannot tell them apart from real runs.

Two knobs matter for testing:
    effect      how strongly the variants differ. 0.0 is the null (no true difference,
                the Friedman test should NOT reject); larger plants a real ranking the
                pipeline must recover.
    seed        makes a generated dataset reproducible.

The generated numbers are deliberately plausible: accuracy near the ~0.35 the probe
reached, efficiency numbers ranked the way the memory probe measured (flash cheapest,
vanilla/sparse dearest). This is so a smoke test of the plots produces something that
looks like the real thing, not so the fake data is mistaken for evidence.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

# The design space, mirrored from make_manifest.py. Kept as a literal rather than
# imported so the generator does not depend on cluster/ being on the path.
DESIGN_SPACE = {
    "d_model": [256, 384],
    "depth": [4, 6],
    "lr": [3e-4, 1e-3],
    "batch_size": [16],
}
VARIANTS = ["vanilla", "flash", "linformer", "linear", "sparse"]
SEEDS = [0, 1, 2]

# Rough per-variant efficiency profiles at the design-space floor, from the memory probe
# and the all-arms run: (sec/epoch, peak_MB). Scaled by depth and d_model below. These
# set the TRUE ranking that a correct pipeline must recover on efficiency metrics.
EFFICIENCY_PROFILE = {
    "flash":     (3.0, 1205),
    "linformer": (3.0, 1462),
    "linear":    (3.3, 1311),
    "sparse":    (7.5, 4205),
    "vanilla":   (10.5, 3949),
}

# True accuracy ordering when effect > 0. Values are offsets in accuracy points added to
# a shared base; flash is the reference at 0. Small and plausible on purpose.
ACCURACY_OFFSET = {
    "flash":     0.010,
    "vanilla":   0.010,   # exact attention: same function as flash, so same accuracy
    "linformer": -0.005,
    "linear":    -0.010,
    "sparse":    -0.020,
}


def build_designs():
    """Enumerate the design space in make_manifest.py's stable order.

    Yields:
        (design_id, design dict) with d_ff derived from d_model.
    """
    keys = sorted(DESIGN_SPACE)
    for i, values in enumerate(itertools.product(*(DESIGN_SPACE[k] for k in keys))):
        design = dict(zip(keys, values))
        design["d_ff"] = 4 * design["d_model"]
        yield i, design


def _param_count(design, variant):
    """Plausible parameter count: a shared base plus linformer's fixed 512,000 overhead."""
    base = 3_678_986 + (design["d_model"] - 256) * 4000 + (design["depth"] - 4) * 300_000
    return base + (512_000 if variant == "linformer" else 0)


def _history(design, variant, effect, rng):
    """Build a plausible 20-epoch history list, matching the per-epoch record schema.

    The final accuracy encodes the planted effect; efficiency is scaled from the profile
    by depth and width. Noise is per-seed so the aggregation stage has something to average.
    """
    depth_scale = design["depth"] / 4
    width_scale = design["d_model"] / 256
    base_sec, base_mb = EFFICIENCY_PROFILE[variant]
    sec = base_sec * depth_scale * width_scale * rng.uniform(0.97, 1.03)
    mb = base_mb * depth_scale * width_scale * rng.uniform(0.99, 1.01)

    base_acc = 0.34 + rng.uniform(-0.01, 0.01)
    final_acc = base_acc + effect * ACCURACY_OFFSET[variant] + rng.gauss(0, 0.004)
    final_acc = max(0.1, min(0.5, final_acc))

    history = []
    for epoch in range(20):
        # A rough climb-then-plateau curve, like the probe's.
        frac = min(1.0, (epoch + 1) / 4)
        acc = 0.14 + (final_acc - 0.14) * frac + rng.gauss(0, 0.006)
        history.append({
            "epoch": epoch,
            "train_loss": 2.4 - 0.7 * frac + rng.gauss(0, 0.01),
            "val_loss": 2.37 - 0.7 * frac + rng.gauss(0, 0.01),
            "val_accuracy": max(0.1, min(0.5, acc)),
            "train_seconds": sec + rng.gauss(0, 0.05),
            "peak_memory_mb": mb,
        })
    return history, final_acc


def make_result(design_id, design, variant, seed, effect, rng):
    """Assemble one result payload identical in shape to train.py's build_result."""
    history, final_acc = _history(design, variant, effect, rng)
    best_acc = max(e["val_accuracy"] for e in history)
    backend = ("flash-attn library varlen FA-2 (sm_86) - VERIFIED"
               if variant == "flash" else None)
    config = {
        "run_id": f"d{design_id:03d}_{variant}_s{seed}",
        "design_id": design_id, "variant": variant, "seed": seed,
        **design, "max_len": 2000, "n_heads": 8, "linformer_k": 256,
        "linformer_sharing": "layerwise", "attn_dropout": 0.0, "dropout": 0.1,
        "sparse_pattern": "fixed", "sparse_stride": 45, "compute_dtype": "bf16",
        "epochs": 20, "task": "listops", "attention": variant,
        "require_flash": variant == "flash",
    }
    return {
        "run_id": config["run_id"],
        "phase": "full",
        "design_id": design_id,
        "variant": variant,
        "seed": seed,
        "config": config,
        "n_params": _param_count(design, variant),
        "backend": backend,
        "best_val_accuracy": best_acc,
        "final_val_accuracy": final_acc,
        "epochs_trained_this_session": 20,
        "mean_train_seconds_per_epoch":
            sum(e["train_seconds"] for e in history) / len(history),
        "peak_memory_mb": max(e["peak_memory_mb"] for e in history),
        "history": history,
    }


def generate(out_dir, effect=1.0, seed=0, drop=None):
    """Write a full synthetic grid to out_dir.

    Args:
        out_dir: directory to write d*.json into.
        effect: 0.0 for the null case; larger plants a stronger true ranking.
        seed: RNG seed for reproducibility.
        drop: optional iterable of run_ids to omit, for testing incomplete-matrix handling.

    Returns:
        The number of files written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drop = set(drop or ())
    rng = random.Random(seed)

    written = 0
    for (design_id, design), variant, s in itertools.product(
        build_designs(), VARIANTS, SEEDS
    ):
        run_id = f"d{design_id:03d}_{variant}_s{s}"
        if run_id in drop:
            continue
        result = make_result(design_id, design, variant, s, effect, rng)
        (out_dir / f"{run_id}.json").write_text(json.dumps(result, indent=2))
        written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RCT results.")
    parser.add_argument("--out", default="results_synth", help="output directory")
    parser.add_argument("--effect", type=float, default=1.0,
                        help="0.0 = null (no true difference); larger = stronger ranking")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop", nargs="*", default=None,
                        help="run_ids to omit, to test incomplete-matrix handling")
    args = parser.parse_args()

    n = generate(args.out, effect=args.effect, seed=args.seed, drop=args.drop)
    print(f"wrote {n} synthetic result files to {args.out}/ "
          f"(effect={args.effect}, seed={args.seed})")


if __name__ == "__main__":
    main()
