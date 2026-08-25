"""Build the run manifest: one CSV row per (design, variant, seed) cell.

The Slurm array index is just an integer, and mapping it to a configuration inside the
job script would put that mapping in bash where it cannot be inspected or diffed. A CSV
written up front lets us see exactly what will run before spending the GPU hours, re-run
a failed task by index into an identical configuration, and join results back to designs
in the analysis without re-deriving anything.

A *design* is a meaningful configuration -- depth, width, lr, batch. It is what we
generalise over and the unit of observation in the Friedman test. A *seed* is noise
control: seeds are averaged within a design x variant cell and are NOT independent
observations, since treating them as such would inflate n and make the statistics wrong.

Every design runs under every variant, because the Demsar pipeline needs a complete
design x variant matrix -- one missing cell drops that whole design from the analysis.
That is why the design space is shared across arms rather than capped per arm.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json

# Order is fixed so array indices stay stable across regenerations.
VARIANTS = ["vanilla", "flash", "linformer", "linear", "sparse"]

# d_model starts at 256, not 128. Linformer's shared E costs a fixed k * max_len =
# 512,000 parameters regardless of depth, which is ~1.14x the total at the floor but
# ~2.95x at d_model=128. Raising the floor closes the gap without modifying any arm,
# without shrinking k (which would handicap the method we are measuring), and without
# breaking the complete matrix. Run count_params.py to regenerate these figures.
DESIGN_SPACE = {
    "d_model": [256, 384],
    "depth": [4, 6],
    "lr": [3e-4, 1e-3],
    "batch_size": [32],
}

SEEDS = [0, 1, 2]

# Held constant -- part of the fixed frame, not the design space.
FIXED = {
    "max_len": 2000,
    "n_heads": 8,
    "linformer_k": 256,
    "linformer_sharing": "layerwise",
    "attn_dropout": 0.0,   # ADR-001
    "dropout": 0.1,
    # "fixed" is Child et al.'s recommended pattern for non-periodic data such as text,
    # and stride 0 means auto -> round(sqrt(seq_len)), their prescription. We record
    # both here rather than leaving them to a code default, so a run is reproducible
    # from the CSV alone.
    "sparse_pattern": "fixed",
    "sparse_stride": 0,
    "epochs": 20,
    "task": "listops",
}

# bigbatch allows 6 running jobs per user.
MAX_CONCURRENT = 6


def build_designs():
    """Enumerate the design space in a stable order.

    Yields:
        (design_id, design dict) pairs, with d_ff derived from d_model.
    """
    keys = sorted(DESIGN_SPACE)
    for i, values in enumerate(itertools.product(*(DESIGN_SPACE[k] for k in keys))):
        design = dict(zip(keys, values))
        design["d_ff"] = 4 * design["d_model"]
        yield i, design


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="cluster/runs.csv")
    args = parser.parse_args()

    designs = list(build_designs())
    rows = []
    for (design_id, design), variant, seed in itertools.product(designs, VARIANTS, SEEDS):
        rows.append({
            "run_id": f"d{design_id:03d}_{variant}_s{seed}",
            "design_id": design_id,
            "variant": variant,
            "seed": seed,
            **design,
            **FIXED,
        })

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n_designs, n_variants, n_seeds = len(designs), len(VARIANTS), len(SEEDS)
    print(f"designs        : {n_designs}")
    print(f"variants       : {n_variants}  {VARIANTS}")
    print(f"seeds          : {n_seeds}")
    print(f"total runs     : {len(rows)}  ({n_designs} x {n_variants} x {n_seeds})")
    print(f"analysis cells : {n_designs * n_variants}  (seeds averaged within cell)")
    print()
    print(f"wrote {args.out}")
    print(f"submit with: sbatch --array=1-{len(rows)}%{MAX_CONCURRENT} "
          "cluster/train_array.slurm")
    print()
    print("design space:")
    print(json.dumps(DESIGN_SPACE, indent=2))


if __name__ == "__main__":
    main()
