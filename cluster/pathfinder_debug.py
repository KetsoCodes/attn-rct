"""Here we separate whether the Pathfinder task is just really hard, or if something is actually broken.

Every variant is sitting at exactly 50% accuracy right now, and the LR warmup probe didn't change that. Being stuck at exactly chance on a 50/50 dataset usually means the model is just guessing or outputting a constant prediction, rather than just learning badly. So here we want to determine if any signal is actually reaching the classifier at all, rather than worrying about which training recipe to use.

This script runs three quick checks to figure this out:

1. DATA: Does the data itself actually carry a usable signal, and what does it look like?
2. CAPACITY: Can the model memorize just 100 examples? If it can't overfit a tiny batch, the wiring is broken, it's not just a tuning issue.
3. THE padding_idx HYPOTHESIS: We want to determine if `padding_idx=0` is permanently zeroing out all the black background pixels (token 0). Since the image is mostly black, this might be destroying the signal before the model can even pool it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

# Ensure the local attn_rct module can be imported when running as a script
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from attn_rct.data import build_dataloaders          # noqa: E402
from attn_rct.model import TransformerClassifier     # noqa: E402

DEFAULT_DATA = "/datasets/fmnisi/pathfinder"


def describe_data(loader, n_batches: int = 40) -> float:
    """Reports what the inputs actually look like and whether the classes differ.

    Args:
        loader: A dataloader over the task.
        n_batches: How many batches to summarise.

    Returns:
        The fraction of tokens equal to zero, which check 3 depends on.
    """
    tokens, labels = [], []
    for i, (ids, _mask, targets) in enumerate(loader):
        if i >= n_batches:
            break
        tokens.append(ids)
        labels.append(targets)
        
    tokens = torch.cat(tokens).numpy()
    labels = torch.cat(labels).numpy()

    zero_fraction = float((tokens == 0).mean())
    print(f"  examples inspected : {len(labels):,}")
    print(f"  sequence length    : {tokens.shape[1]}")
    print(f"  token range        : {tokens.min()} - {tokens.max()}")
    print(f"  distinct tokens    : {len(np.unique(tokens))}")
    print(f"  ZERO-TOKEN FRACTION: {zero_fraction:.1%}")
    
    counts = np.bincount(labels)
    print(f"  class balance      : {dict(enumerate(counts))}")

    # Every example identical would make the task unlearnable for a trivial reason.
    unique_rows = len({row.tobytes() for row in tokens})
    print(f"  distinct images    : {unique_rows:,} of {len(tokens):,}")

    # Pathfinder is built so simple statistics do NOT separate the classes; 
    # a tiny difference here is expected.
    for label in sorted(set(labels.tolist())):
        subset = tokens[labels == label]
        print(f"  class {label}: mean px {subset.mean():7.3f}   "
              f"bright px/img {float((subset > 0).sum(axis=1).mean()):8.1f}")

    if zero_fraction > 0.5:
        print(f"\n  NOTE: {zero_fraction:.0%} of every image is the zero token, which the "
              "frame\n        pins to a non-trainable zero vector via padding_idx=0. "
              "Check 3 tests\n        whether that is what is blocking learning.")
              
    return zero_fraction


def build_model(cfg: SimpleNamespace, vocab_size: int, n_classes: int, 
                unfreeze_zero: bool, device: torch.device) -> nn.Module:
    """Builds the classifier, optionally releasing the token-0 embedding row.

    Args:
        unfreeze_zero: If True, clears padding_idx and gives row 0 a normal 
            initialisation so the background token can carry content.
    """
    torch.manual_seed(0)
    model = TransformerClassifier(cfg, vocab_size=vocab_size, n_classes=n_classes)
    
    if unfreeze_zero:
        model.token_embedding.padding_idx = None
        with torch.no_grad():
            model.token_embedding.weight[0].normal_(0.0, 0.02)
            
    return model.to(device)


def overfit_test(cfg: SimpleNamespace, batch: tuple, meta: dict, device: torch.device, 
                 steps: int, unfreeze_zero: bool, lr: float = 3e-4) -> tuple[float, float]:
    """Trains on one small fixed batch and reports whether the model can memorise it.

    A model that cannot drive training accuracy well above chance on a hundred examples
    it sees over and over is structurally broken, not just under-tuned.

    Returns:
        (final train accuracy, final loss).
    """
    ids, mask, targets = (t.to(device) for t in batch)
    model = build_model(cfg, meta["vocab_size"], meta["n_classes"], unfreeze_zero, device)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    accuracy = loss_value = float("nan")
    
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, mask)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % max(steps // 6, 1) == 0 or step == steps - 1:
            with torch.no_grad():
                accuracy = (model(ids, mask).argmax(-1) == targets).float().mean().item()
            loss_value = loss.item()
            print(f"    step {step:>4}: loss {loss_value:.4f}  train acc {accuracy:.3f}")
            
    return accuracy, loss_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--task", default="pathfinder")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--variant", default="vanilla",
                        help="Exact attention by default: this tests the frame, not an arm")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 74)
    print(f"Pathfinder diagnostic | task={args.task} | device={device}")
    print("=" * 74)

    train_loader, _val_loader, meta = build_dataloaders(
        args.task, args.data_dir, args.max_len,
        batch_size=args.n_examples, seed=0, num_workers=0,
    )
    print(f"meta: {meta}\n")

    print("--- 1. DATA ------------------------------------------------------------")
    zero_fraction = describe_data(train_loader)

    batch = next(iter(train_loader))
    
    cfg = SimpleNamespace(
        d_model=args.d_model, n_heads=8, depth=args.depth, d_ff=4 * args.d_model,
        dropout=0.0,              # No regularisation: we WANT it to overfit
        attn_dropout=0.0,
        max_len=args.max_len, attention=args.variant, variant=args.variant, seed=0,
        linformer_k=256, linformer_sharing="layerwise",
        sparse_pattern="fixed", sparse_stride=32,
        compute_dtype="fp32",     # fp32 so precision is never the explanation
        require_flash=False,
    )

    print(f"\n--- 2. CAPACITY: memorise {args.n_examples} examples, frame as-is --------")
    baseline_acc, _ = overfit_test(cfg, batch, meta, device, args.steps, unfreeze_zero=False)

    print(f"\n--- 3. SAME TEST, token-0 embedding unfrozen ---------------------------")
    unfrozen_acc, _ = overfit_test(cfg, batch, meta, device, args.steps, unfreeze_zero=True)

    chance = 1.0 / meta["n_classes"]
    
    print("\n" + "=" * 74)
    print(f"frame as-is        : train acc {baseline_acc:.3f}")
    print(f"token-0 unfrozen   : train acc {unfrozen_acc:.3f}")
    print(f"chance             : {chance:.3f}")
    print("=" * 74)

    if baseline_acc > 0.95:
        print("VERDICT: the frame CAN memorise this data, so nothing is structurally")
        print("  broken. Pathfinder failing to generalise is then a real result about")
        print("  the task and the training budget, not a bug.")
    elif unfrozen_acc > 0.95 >= baseline_acc:
        print("VERDICT: the frame cannot memorise as-is, but CAN once the token-0")
        print(f"  embedding is unfrozen. padding_idx=0 is the cause: {zero_fraction:.0%} of")
        print("  every image is token 0, and the frame pinned that row to zero.")
        print("  Fix: build the embedding without padding_idx for fixed-length image")
        print("  tasks, then re-run Pathfinder for all five arms.")
    elif max(baseline_acc, unfrozen_acc) < chance + 0.1:
        print("VERDICT: the frame cannot memorise a hundred examples under either")
        print("  setting. Something is broken upstream of the optimiser -- suspect the")
        print("  data pipeline or label alignment before blaming the task.")
    else:
        print("VERDICT: partial learning. Not an obvious structural break, but not")
        print("  healthy either; worth raising the capacity (depth/width) or the step")
        print("  count before concluding anything about the task.")


if __name__ == "__main__":
    main()