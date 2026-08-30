"""Measure worst-case GPU memory for every arm across the design space.

The pilot lost cells to OOM, and the failures correlated with the arm: vanilla and
sparse died where flash and linear did not. That is not a neutral loss. A design is
only usable if EVERY arm completed it, so an arm-correlated OOM silently deletes whole
designs and leaves behind exactly the designs where the memory-hungry arms happened to
fit -- which biases the comparison in favour of the efficient ones through the
missingness itself.

So the feasible configuration has to be established before the grid runs, not
discovered during it. This probe trains a handful of real steps per (design, arm) at
the worst case the loader can produce -- a full batch of max_len sequences with no
padding to shorten them -- and reports peak allocated and peak reserved memory.

Both numbers matter. Allocated is what the tensors need; reserved is what the caching
allocator holds, and the gap between them is fragmentation, which is severe here
because every batch pads to a different length and so requests a differently sized
score matrix. A run OOMs on reserved, not on allocated.

    python cluster/memory_probe.py                       # whole design space
    python cluster/memory_probe.py --batch-size 16       # try a smaller batch
    python cluster/memory_probe.py --designs 2 3 --arms vanilla sparse
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import socket
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cluster"))

try:
    from attn_rct.attention import ATTENTION_REGISTRY
    from attn_rct.model import TransformerClassifier, count_parameters
except ModuleNotFoundError as err:
    print(f"FATAL: cannot import the attention package ({err}).")
    print(f"  looked in: {REPO_ROOT}")
    sys.exit(1)

from make_manifest import FIXED, VARIANTS, build_designs

GIB = 1024 ** 3
VOCAB_SIZE = 20
N_CLASSES = 10


def build_config(design, variant, seed, overrides):
    """Assembles the run config for one (design, arm) cell.

    Args:
        design: a design dict from build_designs.
        variant: arm name.
        seed: RNG seed, which linformer's shared projection also keys off.
        overrides: dict of fields to force, e.g. batch_size or max_len.

    Returns:
        The config namespace.
    """
    fields = {**FIXED, **design, "attention": variant, "variant": variant, "seed": seed}
    fields.update({k: v for k, v in overrides.items() if v is not None})
    # The flash arm must fail loudly rather than fall back: a fallback kernel has
    # different memory behaviour, so a silent one would make this table wrong.
    fields["require_flash"] = variant == "flash"
    return SimpleNamespace(**fields)


def worst_case_batch(cfg, device):
    """Builds the largest batch the loader could ever hand this config.

    Every sequence is exactly max_len and entirely real tokens: dynamic padding pads to
    the batch maximum, so this is the upper bound, and it is also the worst case for the
    flash arm, whose varlen path has nothing to unpad.
    """
    batch, length = int(cfg.batch_size), int(cfg.max_len)
    tokens = torch.randint(1, VOCAB_SIZE, (batch, length), device=device)
    mask = torch.ones(batch, length, dtype=torch.long, device=device)
    targets = torch.randint(0, N_CLASSES, (batch,), device=device)
    return tokens, mask, targets


def measure(cfg, device, steps):
    """Runs a few real training steps and returns peak memory.

    Args:
        cfg: the run config.
        device: CUDA device.
        steps: measured steps, after one warm-up step.

    Returns:
        Dict with n_params, allocated_gib, reserved_gib and status.
    """
    torch.manual_seed(int(cfg.seed))
    model = TransformerClassifier(cfg, vocab_size=VOCAB_SIZE, n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    criterion = nn.CrossEntropyLoss()
    n_params = count_parameters(model)
    tokens, mask, targets = worst_case_batch(cfg, device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(tokens, mask), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # One warm-up step first: Adam allocates its moment buffers on the first step, and
    # we want those counted in steady state rather than as a one-off spike.
    step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(steps):
        step()
    torch.cuda.synchronize(device)

    return {
        "n_params": n_params,
        "allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
        "status": "ok",
    }


def release():
    """Drops everything between cells so one arm's peak does not leak into the next."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=None, help="override the design space")
    parser.add_argument("--max-len", type=int, default=None, help="override FIXED")
    parser.add_argument("--designs", type=int, nargs="*", default=None,
                        help="design ids to probe (default: all)")
    parser.add_argument("--arms", nargs="*", default=None, choices=sorted(ATTENTION_REGISTRY),
                        help="arms to probe (default: all)")
    parser.add_argument("--steps", type=int, default=3, help="measured steps per cell")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write the table to this JSON file")
    parser.add_argument("--headroom", type=float, default=0.85,
                        help="fraction of card capacity above which a cell is called tight")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device visible. This probe measures GPU memory.")
        sys.exit(1)

    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(device).total_memory / GIB
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")

    print("=" * 78)
    print(f"host       : {socket.gethostname()}")
    print(f"gpu        : {torch.cuda.get_device_name(device)}  ({total:.2f} GiB)")
    print(f"torch      : {torch.__version__}")
    print(f"alloc conf : {alloc_conf or '(unset)'}")
    if "expandable_segments" not in alloc_conf:
        print("  WARNING: expandable_segments is not set. Sequence lengths vary every")
        print("  batch, so the allocator fragments badly without it. Reserved will")
        print("  overstate what a properly configured run needs.")
    print("=" * 78)

    designs = dict(build_designs())
    wanted_designs = args.designs if args.designs else sorted(designs)
    wanted_arms = args.arms if args.arms else VARIANTS
    overrides = {"batch_size": args.batch_size, "max_len": args.max_len}

    rows = []
    for design_id in wanted_designs:
        if design_id not in designs:
            print(f"skipping unknown design {design_id}")
            continue
        design = designs[design_id]
        for variant in wanted_arms:
            cfg = build_config(design, variant, args.seed, overrides)
            label = (f"d{design_id:03d} {variant:<9} d_model={cfg.d_model} "
                     f"depth={cfg.depth} batch={cfg.batch_size} len={cfg.max_len}")
            try:
                result = measure(cfg, device, args.steps)
                print(f"{label}  ->  alloc {result['allocated_gib']:6.2f}  "
                      f"reserved {result['reserved_gib']:6.2f} GiB")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                if "out of memory" not in str(err).lower():
                    raise
                result = {"n_params": None, "allocated_gib": None,
                          "reserved_gib": None, "status": "OOM"}
                print(f"{label}  ->  OOM")
            finally:
                release()

            rows.append({"design_id": design_id, "variant": variant,
                         **{k: getattr(cfg, k) for k in
                            ("d_model", "depth", "lr", "batch_size", "max_len")},
                         **result})

    print("\n" + "=" * 78)
    print(f"{'design':>7} {'arm':<10} {'d_model':>8} {'depth':>6} "
          f"{'alloc':>8} {'reserved':>9}  verdict")
    tight, failed = [], []
    for row in rows:
        if row["status"] == "OOM":
            verdict, failed = "OOM", failed + [row]
            numbers = f"{'-':>8} {'-':>9}"
        else:
            fraction = row["reserved_gib"] / total
            verdict = "tight" if fraction > args.headroom else "ok"
            if verdict == "tight":
                tight.append(row)
            numbers = f"{row['allocated_gib']:8.2f} {row['reserved_gib']:9.2f}"
        print(f"  d{row['design_id']:03d} {row['variant']:<10} {row['d_model']:>8} "
              f"{row['depth']:>6} {numbers}  {verdict}")

    print("=" * 78)
    worst = max((r["reserved_gib"] for r in rows if r["status"] == "ok"), default=0.0)
    print(f"worst reserved across all cells: {worst:.2f} GiB of {total:.2f} "
          f"({worst / total:.0%})")

    if failed:
        cells = ", ".join(f"d{r['design_id']:03d}/{r['variant']}" for r in failed)
        print(f"\nFAIL: {len(failed)} cell(s) OOM: {cells}")
        print("  These designs cannot enter the analysis: the Friedman test needs a")
        print("  complete design x variant matrix, and dropping the failed cells drops")
        print("  the whole design. Lower batch_size or depth until this list is empty.")
    elif tight:
        cells = ", ".join(f"d{r['design_id']:03d}/{r['variant']}" for r in tight)
        print(f"\nMARGINAL: {len(tight)} cell(s) above {args.headroom:.0%} of the card: {cells}")
        print("  These fit here but leave no room for a longer batch or a busier node.")
    else:
        print("\nPASS: every cell fits with headroom. The matrix can be completed at "
              "this configuration.")

    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(device),
            "total_gib": total,
            "torch": torch.__version__,
            "alloc_conf": alloc_conf,
            "steps": args.steps,
            "rows": rows,
        }
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
