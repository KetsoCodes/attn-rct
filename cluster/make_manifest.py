"""Build the run manifest: one CSV row per (task, design, variant, seed) cell.

The Slurm array index is just an integer, and mapping it to a configuration inside the
job script would put that mapping in bash where it cannot be inspected or diffed. A CSV
written up front lets us see exactly what will run before spending the GPU hours, re-run
a failed task by index into an identical configuration, and join results back to designs
in the analysis without re-deriving anything.

A *design* is a meaningful architectural configuration -- depth, width, lr. A *seed* is
noise control: seeds are averaged within a cell and are NOT independent observations. A
*task* is a dataset/objective; the study generalises over tasks as well as designs, so
the same design space runs under every task and the analysis blocks on (task, design).

Every (task, design) runs under every variant, because the Demsar pipeline needs a
complete matrix -- one missing cell drops that whole block from the analysis.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json

# Order is fixed so array indices stay stable across regenerations.
VARIANTS = ["vanilla", "flash", "linformer", "linear", "sparse"]

# d_model starts at 256, not 128. Linformer's shared E costs a fixed k * max_len
# parameters regardless of depth, ~1.14x the total at this floor but ~2.95x at 128.
DESIGN_SPACE = {
    "d_model": [256, 384],
    "depth": [4, 6],
    "lr": [3e-4, 1e-3],
    # batch 16, not 32: at 32 the memory probe put vanilla at 97% of a 24GB card at
    # depth 6 and sparse OOM'd outright. 16 leaves the worst cell at ~53%.
    "batch_size": [16],
}

SEEDS = [0, 1, 2]

# Per-task configuration. Each task fixes its own sequence length and data directory,
# because these are properties of the dataset, not of the design space. The attention
# mechanism is the only thing that varies within a (task, design); everything here is
# held constant across the arms of a cell.
TASKS = {
    "listops": {
        "max_len": 2000,
        "data_dir": "/datasets/fmnisi/lra/listops",
    },
    "cifar": {
        # CIFAR-10 Image task is fixed-length 32x32 = 1024, so nothing is filtered or
        # truncated -- the length-cap that limits ListOps accuracy does not apply.
        "max_len": 1024,
        "data_dir": "/datasets/fmnisi/cifar",
    },
    "pathfinder": {
        # Pathfinder32 is also fixed-length 32x32 = 1024. It is the task where approximate
        # attention is reported to struggle most, so it is the sharpest test of whether the
        # accuracy ranking from the other tasks holds.
        "max_len": 1024,
        "data_dir": "/datasets/fmnisi/pathfinder",
    },
}

# Held constant across every task, design, and arm -- the fixed frame.
FIXED = {
    "n_heads": 8,
    "linformer_k": 256,
    "linformer_sharing": "layerwise",
    "attn_dropout": 0.0,   # ADR-001
    "dropout": 0.1,
    "sparse_pattern": "fixed",
    # 45 = round(sqrt(2000)), resolved against ListOps' max_len rather than the padded
    # batch length. Fixed-length tasks pad nothing, so the stride is inert there anyway.
    "sparse_stride": 45,
    "compute_dtype": "bf16",
    "epochs": 20,
}

# bigbatch allows 6 running / 48 submitted jobs per user.
MAX_CONCURRENT = 6
MAX_SUBMIT = 48


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
    parser.add_argument("--tasks", nargs="*", default=sorted(TASKS),
                        help="which tasks to include (default: all registered)")
    args = parser.parse_args()

    for task in args.tasks:
        if task not in TASKS:
            raise SystemExit(f"unknown task {task!r}; known: {sorted(TASKS)}")

    designs = list(build_designs())
    rows = []
    # Task is the outermost loop so a task's rows are contiguous, which makes it easy to
    # submit or re-run one task's block of array indices.
    for task in args.tasks:
        task_cfg = TASKS[task]
        for (design_id, design), variant, seed in itertools.product(
            designs, VARIANTS, SEEDS
        ):
            rows.append({
                # Task in the run_id keeps result filenames and W&B ids unique across
                # tasks: cifar_d000_flash_s0 never collides with listops_d000_flash_s0.
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
