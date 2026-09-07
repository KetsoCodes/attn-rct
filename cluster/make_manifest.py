"""Generates a CSV manifest mapping Slurm array indices to specific training configurations.

This script ensures every combination of task, architectural design, attention variant,
and random seed is explicitly defined before execution. The resulting manifest allows
for reproducible job submissions and clean data joining during statistical analysis.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json

VARIANTS = ["vanilla", "flash", "linformer", "linear", "sparse"]

DESIGN_SPACE = {
    "d_model": [256, 384],
    "depth": [4, 6],
    "lr": [3e-4, 1e-3],
    "batch_size": [16],
}

SEEDS = [0, 1, 2]

TASKS = {
    "listops": {
        "max_len": 2000,
        "data_dir": "/datasets/fmnisi/lra/listops",
    },
    "cifar": {
        "max_len": 1024,
        "data_dir": "/datasets/fmnisi/cifar",
    },
}

FIXED = {
    "n_heads": 8,
    "linformer_k": 256,
    "linformer_sharing": "layerwise",
    "attn_dropout": 0.0,
    "dropout": 0.1,
    "sparse_pattern": "fixed",
    "sparse_stride": 45,
    "compute_dtype": "bf16",
    "epochs": 20,
}

MAX_CONCURRENT = 6
MAX_SUBMIT = 48


def build_designs():
    """Enumerates the architectural design space in a stable, deterministic order.

    Yields:
        Tuple of (design_id, design_dictionary) where d_ff is dynamically derived.
    """
    keys = sorted(DESIGN_SPACE)
    for i, values in enumerate(itertools.product(*(DESIGN_SPACE[k] for k in keys))):
        design = dict(zip(keys, values))
        design["d_ff"] = 4 * design["d_model"]
        yield i, design


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="cluster/runs.csv")
    parser.add_argument("--tasks", nargs="*", default=sorted(TASKS),
                        help="which tasks to include (default: all registered)")
    args = parser.parse_args()

    for task in args.tasks:
        if task not in TASKS:
            raise SystemExit(f"unknown task {task!r}; known: {sorted(TASKS)}")

    designs = list(build_designs())
    rows = []

    for task in args.tasks:
        task_cfg = TASKS[task]
        for (design_id, design), variant, seed in itertools.product(
            designs, VARIANTS, SEEDS
        ):
            rows.append({
                "run_id": f"{task}_d{design_id:03d}_{variant}_s{seed}",
                "task": task,
                "design_id": design_id,
                "variant": variant,
                "seed": seed,
                "data_dir": task_cfg["data_dir"],
                "max_len": task_cfg["max_len"],
                **design,
                **FIXED,
            })

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n_tasks = len(args.tasks)
    n_designs, n_variants, n_seeds = len(designs), len(VARIANTS), len(SEEDS)
    n_blocks = n_tasks * n_designs
    
    print(f"tasks          : {n_tasks}  {args.tasks}")
    print(f"designs/task   : {n_designs}")
    print(f"variants       : {n_variants}  {VARIANTS}")
    print(f"seeds          : {n_seeds}")
    print(f"total runs     : {len(rows)}  "
          f"({n_tasks} x {n_designs} x {n_variants} x {n_seeds})")
    print(f"analysis blocks: {n_blocks}  (task x design)")
    print(f"analysis cells : {n_blocks * n_variants}  (seeds averaged within cell)")
    print()
    print(f"wrote {args.out}")
    print(f"NOTE: {len(rows)} runs exceeds the {MAX_SUBMIT}-job submit limit; "
          f"submit in blocks of <= {MAX_SUBMIT}, e.g.")
    print(f"  sbatch --array=1-{MAX_SUBMIT}%{MAX_CONCURRENT} cluster/train_array.slurm")
    print(f"  sbatch --array={MAX_SUBMIT + 1}-{2 * MAX_SUBMIT}%{MAX_CONCURRENT} ...")


if __name__ == "__main__":
    main()