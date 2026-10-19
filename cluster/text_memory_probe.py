"""Profiles memory and training cost for the Text task before submitting the full grid.

Text uses sequences up to 4000 tokens, meaning Vanilla attention requires a 4000x4000 
score matrix per head. A batch of 16 won't fit in memory, so it must be split into 
micro-batches. This script determines the largest micro-batch that fits the most 
memory-hungry arm without OOMing, then estimates the total GPU hours for the full run.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from attn_rct.data import load_text                      # noqa: E402
from attn_rct.model import TransformerClassifier         # noqa: E402

VARIANTS = ["vanilla", "flash", "linformer", "linear", "sparse"]
DESIGNS = {"small (d256 L4)": (256, 4), "large (d384 L6)": (384, 6)}
MAX_LEN = 4000
N_TRAIN = 22_500            
EPOCHS, SEEDS, N_DESIGNS = 20, 3, 8


def make_cfg(arm, d_model, depth):
    return SimpleNamespace(
        d_model=d_model, n_heads=8, depth=depth, d_ff=4 * d_model,
        dropout=0.1, attn_dropout=0.0, max_len=MAX_LEN,
        attention=arm, variant=arm, seed=0,
        linformer_k=256, linformer_sharing="layerwise",
        sparse_pattern="fixed", sparse_stride=45,
        compute_dtype="bf16", require_flash=(arm == "flash"),
    )


def build(arm, d_model, depth, device):
    torch.manual_seed(0)
    model = TransformerClassifier(make_cfg(arm, d_model, depth), vocab_size=256, n_classes=2)
    return model.to(device), torch.optim.Adam(model.parameters(), lr=1e-3)


def worst_case_peak(arm, micro, device):
    """Measures peak memory usage on the largest design with full 4000-token reviews."""
    model, opt = build(arm, 384, 6, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    try:
        tokens = torch.randint(1, 256, (micro, MAX_LEN), device=device)
        mask = torch.ones(micro, MAX_LEN, dtype=torch.long, device=device)
        targets = torch.randint(0, 2, (micro,), device=device)
        
        opt.zero_grad(set_to_none=True)
        nn.CrossEntropyLoss()(model(tokens, mask), targets).backward()
        opt.step()
        
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 1e6
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del model, opt
        torch.cuda.empty_cache()


def real_seconds_per_epoch(arm, d_model, depth, loader, micro, device, n_micro=48):
    """Times real data batches to estimate epoch duration without overestimating on padding."""
    model, opt = build(arm, d_model, depth, device)
    criterion = nn.CrossEntropyLoss()
    accum = 16 // micro
    batches = iter(loader)
    
    # Warm up kernels
    for _ in range(4):
        tokens, mask, targets = (t.to(device) for t in next(batches))
        (criterion(model(tokens, mask), targets) / accum).backward()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    
    start = time.time()
    for i in range(n_micro):
        tokens, mask, targets = (t.to(device) for t in next(batches))
        (criterion(model(tokens, mask), targets) / accum).backward()
        
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            
    torch.cuda.synchronize(device)
    per_micro = (time.time() - start) / n_micro
    
    del model, opt
    torch.cuda.empty_cache()
    return per_micro * (N_TRAIN // micro)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/datasets/fmnisi/text")
    parser.add_argument("--micro", type=int, nargs="+", default=[8, 4])
    parser.add_argument("--concurrent", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda")
    total_mb = torch.cuda.get_device_properties(device).total_memory / 1e6
    print(f"GPU: {torch.cuda.get_device_name(device)}, {total_mb:,.0f} MB")

    # 1. Memory, worst case
    print("\n--- 1. WORST-CASE MEMORY: largest design, full 4000-token reviews ---")
    print(f"{'arm':<10s}" + "".join(f"{'micro ' + str(m):>16s}" for m in args.micro))
    fits = {m: True for m in args.micro}
    
    for arm in VARIANTS:
        cells = []
        for m in args.micro:
            peak = worst_case_peak(arm, m, device)
            if peak is None:
                cells.append(f"{'OOM':>16s}")
                fits[m] = False
            else:
                share = peak / total_mb
                if share > 0.85:
                    fits[m] = False
                cells.append(f"{peak:>9,.0f} ({share:>3.0%})")
        print(f"{arm:<10s}" + "".join(cells))

    usable = [m for m in args.micro if fits[m]]
    if not usable:
        print("\nNo tested micro-batch fits every arm within 85% of the card. Re-run with --micro 2.")
        return
        
    micro = max(usable)
    print(f"\n=> micro-batch {micro} fits every arm (<= 85% of the card). grad_accum_steps = {16 // micro}.")

    # 2. Cost, on real batches
    print(f"\n--- 2. COST on real review batches, micro-batch {micro} ---")
    train_loader, _, _ = load_text(args.data_dir, MAX_LEN, batch_size=micro, seed=0, num_workers=0)
    total_seconds = 0.0
    
    print(f"{'arm':<10s}" + "".join(f"{name:>20s}" for name in DESIGNS))
    for arm in VARIANTS:
        per = []
        for d_model, depth in DESIGNS.values():
            per.append(real_seconds_per_epoch(arm, d_model, depth, train_loader, micro, device))
        print(f"{arm:<10s}" + "".join(f"{s:>14,.0f} s/ep" for s in per))
        total_seconds += (sum(per) / len(per)) * EPOCHS * SEEDS * N_DESIGNS

    gpu_h = total_seconds / 3600
    wall_days = gpu_h / args.concurrent / 24
    
    print(f"\nEstimated full Text grid (8 designs x 5 arms x 3 seeds x {EPOCHS} epochs):")
    print(f"  {gpu_h:,.0f} GPU-hours  ~  {wall_days:.1f} days at {args.concurrent} concurrent")
    print(f"  4-design fallback: {gpu_h / 2:,.0f} GPU-hours  ~  {wall_days / 2:.1f} days")


if __name__ == "__main__":
    main()