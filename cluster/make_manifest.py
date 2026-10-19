"""Build the run manifest: one CSV row per (task, design, variant, seed) cell.

Writing the configuration to a CSV upfront lets me see exactly what will run before
spending GPU hours. It also makes it easy to re-run failed jobs by index and join
results back for analysis.

- Design: Architectural config (depth, width, lr).
- Seed: Noise control (averaged within a cell, not independent).
- Task: Dataset/objective. The same design space runs under every task.
- Variant: The attention mechanism. The analysis requires a complete matrix.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json

# Fixed order to keep array indices stable across regenerations
VARIANTS = ["vanilla", "flash", "linformer", "linear", "sparse"]

# d_model starts at 256. Linformer's shared E projection costs a fixed k * max_len
# parameters, which would dominate the total model size at 128.
DESIGN_SPACE = {
    "d_model": [256, 384],
    "depth": [4, 6],
    "lr": [3e-4, 1e-3],
    # Batch size 16. At 32, Vanilla used 97% of a 24GB card and Sparse OOM'd.
    "batch_size": [16],
}

SEEDS = [0, 1, 2]

# Task-specific settings. Everything else is held constant across arms.
TASKS = {
    "listops": {
        "max_len": 2000,
        "data_dir": "/datasets/fmnisi/lra/listops",
    },
    "cifar": {
        # CIFAR-10 is fixed-length (32x32 = 1024), so no filtering or truncation.
        "max_len": 1024,
        "data_dir": "/datasets/fmnisi/cifar",
    },
    "pathfinder": {
        # Pathfinder32 is also fixed at 1024. This is the sharpest test for approximate attention.
        "max_len": 1024,
        "data_dir": "/datasets/fmnisi/pathfinder",
    },
    "text": {
        # IMDb sentiment capped at 4000 tokens. Vanilla attention can't fit a batch
        # of 16 in memory, so I use gradient accumulation. The micro-batch size is
        # set by the most memory-hungry arm and applied to all five for fair throughput.
        "max_len": 4000,
        "data_dir": "/datasets/fmnisi/text",
        "grad_accum_steps": 4,
    },
}

# The fixed frame: held constant across every run.
FIXED = {
    "n_heads": 8,
    "linformer_k": 256,
    "linformer_sharing": "layerwise",
    "attn_dropout": 0.0,
    "dropout": 0.1,
    "sparse_pattern": "fixed",
    # 45 = round(sqrt(2000)), sized for ListOps. Fixed-length tasks ignore this.
    "sparse_stride": 45,
    "compute_dtype": "bf16",
    "epochs": 20,
}

# bigbatch partition limits
MAX_CONCURRENT = 6
MAX_SUBMIT = 48


def build_designs():
    """Enumerates the design space in a stable order."""
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
    
    # Task as the outermost loop groups rows nicely for targeted resubmissions.
    for task in args.tasks:
        task_cfg = TASKS[task]
        for (design_id, design), variant, seed in itertools.product(
            designs, VARIANTS, SEEDS
        ):
            rows.append({
                # Include task in run_id to prevent collisions across datasets
                "run_id": f"{task}_d{design_id:03d}_{variant}_s{seed}",
                "task": task,
                "design_id": design_id,
                "variant": variant,
                "seed": seed,
                "data_dir": task_cfg["data_dir"],
                "max_len": task_cfg["max_len"],
                # Default to 1 so older rows behave identically
                "grad_accum_steps": task_cfg.get("grad_accum_steps", 1),
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