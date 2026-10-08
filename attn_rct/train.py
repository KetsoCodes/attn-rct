"""This is my training entry point for a single (design, variant, seed) experimental cell.

It runs one training session based on the Slurm array index mapped to my manifest. 
I included periodic checkpointing and clean signal handling so runs survive cluster 
preemptions, which keeps the RNG state intact for strict experimental pairing.

It finishes by writing a single JSON file with the averaged metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from .data import build_dataloaders
from .model import TransformerClassifier, count_parameters

INTERRUPTED = False
EXIT_REQUEUE = 42


def handle_signal(signum, frame):
    """Catches Slurm walltime warnings or kill signals so I can trigger a clean exit."""
    global INTERRUPTED
    print(f"\n[signal {signum}] walltime approaching -- checkpointing and exiting",
          flush=True)
    INTERRUPTED = True


def parse_args():
    """Parses command-line arguments and my configuration overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--data-dir", default=None,
                        help="overrides the manifest's per-task data_dir if given")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--checkpoint-every-min", type=float, default=30.0)
    parser.add_argument("--require-flash", action="store_true",
                        help="fail loudly if the true FA-2 path is unreachable")
    parser.add_argument("--epochs", type=int, default=None, help="override the manifest")
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="smoke-test only: cap batches per epoch")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--phase", default="pilot", choices=["smoke", "pilot", "full"],
                        help="Categorises runs to separate throwaway tests from reportable data")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="override the manifest")
    parser.add_argument("--compute-dtype", default=None,
                        choices=["fp32", "bf16", "fp16", "auto"],
                        help="override the manifest's attention compute precision; "
                             "note the flash arm's kernels accept only fp16/bf16")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="linear LR warmup over this many steps; 0 disables it")
    return parser.parse_args()


def load_manifest_row(manifest: Path, row_index: int) -> SimpleNamespace:
    """Grabs a specific row from the manifest CSV and casts the numbers properly."""
    with open(manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 1 <= row_index <= len(rows):
        raise IndexError(f"row {row_index} out of range; manifest has {len(rows)} rows")

    raw = rows[row_index - 1]
    typed = {}
    for key, value in raw.items():
        for cast in (int, float):
            try:
                typed[key] = cast(value)
                break
            except (ValueError, TypeError):
                continue
        else:
            typed[key] = value
    return SimpleNamespace(**typed)


def resolve_config(args) -> SimpleNamespace:
    """Combines the manifest config with any command-line overrides I pass in."""
    cfg = load_manifest_row(Path(args.manifest), args.row)
    cfg.attention = cfg.variant

    if args.epochs is not None:
        cfg.epochs = args.epochs
    for name in ("batch_size", "max_len", "d_model", "depth", "lr", "compute_dtype"):
        override = getattr(args, name)
        if override is not None:
            print(f"[override] {name}: {getattr(cfg, name, None)} -> {override}")
            setattr(cfg, name, override)
    if args.d_model is not None:
        cfg.d_ff = 4 * cfg.d_model

    # I leave warmup OFF by default so every run made before it existed is reproduced exactly.
    if args.warmup_steps is not None:
        print(f"[override] warmup_steps: "
              f"{getattr(cfg, 'warmup_steps', 0)} -> {args.warmup_steps}")
        cfg.warmup_steps = args.warmup_steps
    elif not hasattr(cfg, "warmup_steps"):
        cfg.warmup_steps = 0

    cfg.require_flash = args.require_flash
    return cfg


def apply_warmup(optimizer, base_lr: float, global_step: int, warmup_steps: int) -> float:
    """Linearly ramps the learning rate from 0 to base_lr over the first warmup_steps.

    Without a warmup, Transformers on tasks like Pathfinder just sit at chance forever.
    The first massive updates hit before the attention mechanism has time to organize,
    and the model never recovers. Adding a plain linear ramp (like the original LRA recipe)
    is the smallest change I can make to test that hypothesis.

    If warmup_steps is 0, the optimizer is untouched, meaning my older runs stay perfectly 
    reproducible.

    Args:
        optimizer: the optimizer whose param groups carry the learning rate.
        base_lr: the configured learning rate, the value warmup ramps up to.
        global_step: steps completed since the start of training, not of this session.
        warmup_steps: length of the ramp; 0 disables it.

    Returns:
        The learning rate now in effect, for logging.
    """
    if not warmup_steps:
        return base_lr
    scale = min(1.0, (global_step + 1) / float(warmup_steps))
    lr = base_lr * scale
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def serialisable_config(cfg) -> dict:
    """Strips out private attributes so I can dump the config to JSON safely."""
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


def evaluate(model, loader, device, limit_batches=None):
    """Runs the model against the validation set."""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (tokens, mask, targets) in enumerate(loader):
            if limit_batches and i >= limit_batches:
                break
            tokens, mask, targets = tokens.to(device), mask.to(device), targets.to(device)
            logits = model(tokens, mask)
            loss_sum += criterion(logits, targets).item() * targets.size(0)
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            total += targets.size(0)
    model.train()
    return correct / max(total, 1), loss_sum / max(total, 1)


def save_checkpoint(path, model, optimizer, epoch, best_accuracy):
    """Writes a resumable checkpoint atomically so I don't corrupt files if it crashes mid-write."""
    tmp = Path(str(path) + ".tmp")
    torch.save({
        "epoch": epoch,
        "best_accuracy": best_accuracy,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, tmp)
    tmp.replace(path)
    print(f"[checkpoint] epoch {epoch} -> {path}", flush=True)


def resume(checkpoint_path, model, optimizer, device):
    """Loads the model weights, optimizer state, and RNG state if I have a checkpoint."""
    if not checkpoint_path.exists():
        return 0, 0.0

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"].cpu())
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])

    start_epoch = state["epoch"] + 1
    best_accuracy = state["best_accuracy"]
    print(f"[resume] from epoch {start_epoch}, best acc {best_accuracy:.4f}")
    return start_epoch, best_accuracy


def start_wandb(args, cfg, n_params, vocab_size):
    """Fires up Weights & Biases logging if my API key is set."""
    enabled = (os.environ.get("WANDB_API_KEY")
               or os.environ.get("WANDB_MODE") == "offline")
    if not enabled:
        return None

    try:
        import wandb
        return wandb.init(
            project=os.environ.get("WANDB_PROJECT", "attn-rct"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=f"{args.phase}/{cfg.run_id}",
            id=f"{args.phase}-{cfg.run_id}",
            group=f"{args.phase}/{cfg.task}/design_{cfg.design_id}",
            job_type=cfg.variant,
            tags=[args.phase, cfg.task, cfg.variant,
                  f"seed{cfg.seed}", f"design{cfg.design_id}"],
            config={**serialisable_config(cfg), "phase": args.phase,
                    "n_params": n_params, "vocab_size": vocab_size},
            resume="allow",
        )
    except Exception as err:
        print(f"[wandb] disabled: {err!r}")
        return None


def write_json_atomic(path: Path, payload: dict):
    """Writes the final JSON payload atomically to prevent corrupted files on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def build_result(cfg, args, history, n_params, backend, best_accuracy) -> dict:
    """Pulls together the final results payload for my analysis script."""
    latest = history[-1] if history else {}
    return {
        "run_id": cfg.run_id,
        "phase": args.phase,
        "design_id": cfg.design_id,
        "variant": cfg.variant,
        "seed": cfg.seed,
        "config": serialisable_config(cfg),
        "n_params": n_params,
        "backend": backend,
        "best_val_accuracy": best_accuracy,
        "final_val_accuracy": latest.get("val_accuracy"),
        "epochs_trained_this_session": len(history),
        "mean_train_seconds_per_epoch":
            sum(e["train_seconds"] for e in history) / len(history) if history else None,
        "peak_memory_mb": max((e.get("peak_memory_mb", 0) for e in history), default=None),
        "nonfinite_grad_steps": latest.get("nonfinite_grad_steps", 0),
        "history": history,
    }


def main():
    args = parse_args()
    signal.signal(signal.SIGUSR1, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    cfg = resolve_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== {cfg.run_id} | variant={cfg.variant} | seed={cfg.seed} | {device} ===")
    print(f"config: {vars(cfg)}")

    data_dir = args.data_dir or getattr(cfg, "data_dir", None)
    if data_dir is None:
        raise SystemExit(
            "no data directory: pass --data-dir or include data_dir in the manifest"
        )
    train_loader, val_loader, meta = build_dataloaders(
        cfg.task, data_dir, cfg.max_len, cfg.batch_size, cfg.seed, args.num_workers,
    )
    print(f"task: {cfg.task} | vocab: {meta['vocab_size']} | "
          f"classes: {meta['n_classes']} | train batches: {len(train_loader)} "
          f"| val batches: {len(val_loader)}")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    model = TransformerClassifier(
        cfg, vocab_size=meta["vocab_size"], n_classes=meta["n_classes"]
    ).to(device)
    n_params = count_parameters(model)
    print(f"parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = Path(args.checkpoint_dir) / "last.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch, best_accuracy = resume(checkpoint_path, model, optimizer, device)

    run = start_wandb(args, cfg, n_params, meta["vocab_size"])

    backend = None
    if cfg.variant == "flash" and torch.cuda.is_available():
        from .attention import FlashAttention
        backend = FlashAttention.measure_backend(device)
        print(f"[backend] {backend}")

    if start_epoch >= int(cfg.epochs):
        print(f"WARNING: checkpoint is at epoch {start_epoch} and target is "
              f"{cfg.epochs} -- nothing to train. Timing/memory will be null. "
              f"Delete {checkpoint_path.parent} to force a fresh run.")

    last_checkpoint = time.time()
    history = []
    
    # I count this across the whole session, not per epoch. If a run silently stops learning,
    # I want one single number my analysis script can screen for.
    nonfinite_grad_steps = 0
    
    # Warmup is tied to the global step. If I resume a run, it continues the ramp from where
    # it left off instead of restarting it (which would just re-apply tiny updates).
    base_lr = float(cfg.lr)
    warmup_steps = int(getattr(cfg, "warmup_steps", 0) or 0)
    batches_per_epoch = len(train_loader)
    current_lr = base_lr
    if warmup_steps:
        print(f"[warmup] linear ramp to lr={base_lr} over {warmup_steps} steps "
              f"({warmup_steps / max(batches_per_epoch, 1):.2f} epochs)")

    for epoch in range(start_epoch, int(cfg.epochs)):
        model.train()
        epoch_start = time.time()
        running_loss, seen = 0.0, 0

        for i, (tokens, mask, targets) in enumerate(train_loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            tokens, mask, targets = tokens.to(device), mask.to(device), targets.to(device)

            current_lr = apply_warmup(
                optimizer, base_lr, epoch * batches_per_epoch + i, warmup_steps
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens, mask), targets)
            loss.backward()
            
            # I left the clipping math exactly as it was, so every result I ran before adding 
            # this counter is perfectly reproducible. I'm just looking at the return value now.
            #
            # Here's why I'm counting it: `clip_grad_norm_` scales gradients by max_norm/total_norm.
            # If the total_norm is infinite, the coefficient becomes 1/inf = 0, meaning the optimizer 
            # applies an update of exactly zero. The loss freezes, but training happily reports 
            # success instead of throwing a NaN or crashing. That's exactly what trapped my 1e-2 
            # Pathfinder runs. Since PyTorch is silently failing here, I need to count these 
            # non-finite steps and track them with the results.
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(total_norm):
                nonfinite_grad_steps += 1
                if nonfinite_grad_steps in (1, 10, 100, 1000):
                    print(f"WARNING: non-finite gradient norm at epoch {epoch} step {i} "
                          f"({nonfinite_grad_steps} so far). Clipping scales this update "
                          f"to zero, so the model is not learning from these steps.",
                          flush=True)
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            seen += targets.size(0)

            if INTERRUPTED:
                save_checkpoint(checkpoint_path, model, optimizer, epoch - 1, best_accuracy)
                if run:
                    run.finish()
                print(f"checkpointed on signal at epoch {epoch}; "
                      f"exiting {EXIT_REQUEUE} to request requeue")
                sys.exit(EXIT_REQUEUE)

            if (time.time() - last_checkpoint) / 60 > args.checkpoint_every_min:
                save_checkpoint(checkpoint_path, model, optimizer, epoch - 1, best_accuracy)
                last_checkpoint = time.time()

        train_time = time.time() - epoch_start
        val_accuracy, val_loss = evaluate(model, val_loader, device, args.limit_batches)
        best_accuracy = max(best_accuracy, val_accuracy)

        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "train_seconds": train_time,
            "lr": current_lr,
            "nonfinite_grad_steps": nonfinite_grad_steps,
        }
        if torch.cuda.is_available():
            record["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
        history.append(record)
        print(f"epoch {epoch}: {record}", flush=True)
        if run:
            run.log(record)

        save_checkpoint(checkpoint_path, model, optimizer, epoch, best_accuracy)
        last_checkpoint = time.time()

    result = build_result(cfg, args, history, n_params, backend, best_accuracy)
    write_json_atomic(Path(args.result_file), result)
    print(f"[result] {args.result_file}")

    if run:
        run.summary["best_val_accuracy"] = best_accuracy
        run.finish()


if __name__ == "__main__":
    main()