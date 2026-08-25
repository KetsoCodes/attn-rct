"""Training entry point for a single (design, variant, seed) experimental cell.

Executes a single training run where the Slurm array index maps to a specific configuration 
in the manifest. Includes periodic checkpointing and clean signal handling to survive 
cluster preemption, ensuring RNG state is preserved for strict experimental pairing.

Writes a single JSON file containing averaged metrics.
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
    """Flags when a Slurm walltime warning or kill signal is received to trigger a clean exit."""
    global INTERRUPTED
    print(f"\n[signal {signum}] walltime approaching -- checkpointing and exiting",
          flush=True)
    INTERRUPTED = True


def parse_args():
    """Parses command-line arguments and configuration overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--data-dir", required=True)
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
    return parser.parse_args()


def load_manifest_row(manifest: Path, row_index: int) -> SimpleNamespace:
    """Reads a specific row from the manifest CSV and casts numerical fields appropriately."""
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
    """Combines the manifest row configuration with any command-line overrides."""
    cfg = load_manifest_row(Path(args.manifest), args.row)
    cfg.attention = cfg.variant

    if args.epochs is not None:
        cfg.epochs = args.epochs
    for name in ("batch_size", "max_len", "d_model", "depth"):
        override = getattr(args, name)
        if override is not None:
            print(f"[override] {name}: {getattr(cfg, name)} -> {override}")
            setattr(cfg, name, override)
    if args.d_model is not None:
        cfg.d_ff = 4 * cfg.d_model

    cfg.require_flash = args.require_flash
    return cfg


def serialisable_config(cfg) -> dict:
    """Returns a JSON-safe dictionary of the configuration, filtering out private attributes."""
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


def evaluate(model, loader, device, limit_batches=None):
    """Evaluates the model on the validation dataset."""
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
    """Writes a resumable checkpoint atomically via a temporary file."""
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
    """Restores model weights, optimizer state, and RNG state if a checkpoint exists."""
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
    """Initialises Weights & Biases logging if credentials are provided."""
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
            group=f"{args.phase}/design_{cfg.design_id}",
            job_type=cfg.variant,
            tags=[args.phase, cfg.variant, f"seed{cfg.seed}", f"design{cfg.design_id}"],
            config={**serialisable_config(cfg), "phase": args.phase,
                    "n_params": n_params, "vocab_size": vocab_size},
            resume="allow",
        )
    except Exception as err:
        print(f"[wandb] disabled: {err!r}")
        return None


def write_json_atomic(path: Path, payload: dict):
    """Writes a JSON payload atomically to prevent corrupted files on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def build_result(cfg, args, history, n_params, backend, best_accuracy) -> dict:
    """Assembles the final results payload for later analysis."""
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

    train_loader, val_loader, vocab = build_dataloaders(
        args.data_dir, cfg.max_len, cfg.batch_size, cfg.seed, args.num_workers,
    )
    print(f"vocab: {len(vocab)} | train batches: {len(train_loader)} "
          f"| val batches: {len(val_loader)}")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    model = TransformerClassifier(cfg, vocab_size=len(vocab)).to(device)
    n_params = count_parameters(model)
    print(f"parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = Path(args.checkpoint_dir) / "last.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch, best_accuracy = resume(checkpoint_path, model, optimizer, device)

    run = start_wandb(args, cfg, n_params, len(vocab))

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
    
    for epoch in range(start_epoch, int(cfg.epochs)):
        model.train()
        epoch_start = time.time()
        running_loss, seen = 0.0, 0

        for i, (tokens, mask, targets) in enumerate(train_loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            tokens, mask, targets = tokens.to(device), mask.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens, mask), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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