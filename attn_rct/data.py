"""Task data pipelines behind a unified registry.

Standardizes different datasets like ListOps, CIFAR-10.... into a consistent format
for the model: a padded batch of token IDs, a padding mask, and integer labels.
This allows new tasks to be added seamlessly without modifying the training loop.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def collate(batch):
    """Pads a batch of (sequences, targets) to the longest sequence in the batch.
    """
    sequences, targets = zip(*batch)
    longest = max(len(s) for s in sequences)
    padded = torch.zeros(len(sequences), longest, dtype=torch.long)
    mask = torch.zeros(len(sequences), longest, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1
    return padded, mask, torch.tensor(targets, dtype=torch.long)


def _make_loaders(train_set, val_set, batch_size, seed, num_workers):
    """Builds the train/val DataLoaders with seeded, reproducible batch ordering."""
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=generator,
        collate_fn=collate, num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers,
    )
    return train_loader, val_loader


# ---- ListOps dataset loader (there's some modifcation need due then variable length)

def read_tsv(path: Path) -> list[tuple[str, int]]:
    """Reads a ListOps TSV file, returning (source, target) pairs.

    Raises:
        ValueError: If the file is not in the expected Source/Target schema.
    """
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "Source" not in reader.fieldnames or "Target" not in reader.fieldnames:
            raise ValueError(
                f"{path} has columns {reader.fieldnames}; expected Source/Target."
            )
        for row in reader:
            rows.append((row["Source"], int(row["Target"])))
    return rows


def build_listops_vocab(rows) -> dict[str, int]:
    """Generates a deterministic vocabulary from the ListOps training split."""
    tokens = set()
    for source, _ in rows:
        tokens.update(source.split())
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in sorted(tokens):
        vocab[token] = len(vocab)
    return vocab


class ListOpsDataset(Dataset):
    """Tokenizes ListOps examples and filters out sequences longer than max_len."""

    def __init__(self, rows, vocab, max_len):
        self.vocab = vocab
        self.max_len = max_len
        self.examples = []
        self.n_filtered = 0
        for source, target in rows:
            ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in source.split()]
            if len(ids) > max_len:
                self.n_filtered += 1
                continue
            self.examples.append((ids, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ids, target = self.examples[index]
        return torch.tensor(ids, dtype=torch.long), target


def load_listops(data_dir, max_len, batch_size, seed, num_workers=2):
    """Loads and processes the ListOps dataset."""
    data_dir = Path(data_dir)
    files = {split: data_dir / f"basic_{split}.tsv" for split in ("train", "val")}
    for split, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {path}. Expected basic_train/val.tsv layout.")

    train_rows = read_tsv(files["train"])
    val_rows = read_tsv(files["val"])
    vocab = build_listops_vocab(train_rows)

    train_set = ListOpsDataset(train_rows, vocab, max_len)
    val_set = ListOpsDataset(val_rows, vocab, max_len)

    if train_set.n_filtered or val_set.n_filtered:
        kept = len(train_set) / max(len(train_rows), 1)
        print(f"NOTE: filtered {train_set.n_filtered} train and {val_set.n_filtered} val "
              f"sequences longer than max_len={max_len}. {kept:.0%} of train kept.")

    train_loader, val_loader = _make_loaders(
        train_set, val_set, batch_size, seed, num_workers
    )
    meta = {"vocab_size": len(vocab), "n_classes": 10, "max_len": max_len}
    return train_loader, val_loader, meta


# ---- CIFAR-10, here the length is fixed.

CIFAR_SEQ_LEN = 1024
CIFAR_VOCAB = 256
CIFAR_CLASSES = 10
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _load_cifar_batch(path: Path):
    """Reads a CIFAR-10 pickle batch file."""
    with open(path, "rb") as handle:
        entry = pickle.load(handle, encoding="bytes")
    return entry[b"data"], entry[b"labels"]


def cifar_to_grey_tokens(data: np.ndarray) -> np.ndarray:
    """Converts raw CIFAR RGB arrays into 1D sequences of 8-bit greyscale tokens.
    """
    n = data.shape[0]
    rgb = data.reshape(n, 3, CIFAR_SEQ_LEN)
    grey = np.tensordot(_LUMA, rgb.astype(np.float64), axes=([0], [1]))
    return np.clip(np.round(grey), 0, 255).astype(np.uint8)


class CifarDataset(Dataset):
    """Dataset for greyscale-flattened CIFAR-10 images as fixed-length token sequences."""

    def __init__(self, tokens: np.ndarray, labels):
        self.tokens = tokens
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ids = torch.from_numpy(self.tokens[index].astype(np.int64))
        return ids, int(self.labels[index])


def load_cifar(data_dir, max_len, batch_size, seed, num_workers=2, val_fraction=0.1):
    """Loads and processes the CIFAR-10 dataset.
    """
    if max_len != CIFAR_SEQ_LEN:
        raise ValueError(
            f"CIFAR-10 is a fixed-length task at {CIFAR_SEQ_LEN} tokens; "
            f"max_len={max_len} does not match. Set max_len={CIFAR_SEQ_LEN} for this task."
        )

    root = Path(data_dir) / "cifar-10-batches-py"
    if not root.exists():
        raise FileNotFoundError(
            f"missing {root}. Download and extract cifar-10-python.tar.gz there."
        )

    data_parts, label_parts = [], []
    for i in range(1, 6):
        part = root / f"data_batch_{i}"
        if not part.exists():
            raise FileNotFoundError(f"missing {part}")
        data, labels = _load_cifar_batch(part)
        data_parts.append(data)
        label_parts.extend(labels)

    data = np.concatenate(data_parts, axis=0)
    tokens = cifar_to_grey_tokens(data)
    labels = np.array(label_parts)

    rng = np.random.default_rng(12345)
    order = rng.permutation(len(labels))
    n_val = int(len(labels) * val_fraction)
    val_idx, train_idx = order[:n_val], order[n_val:]

    train_set = CifarDataset(tokens[train_idx], labels[train_idx])
    val_set = CifarDataset(tokens[val_idx], labels[val_idx])

    train_loader, val_loader = _make_loaders(
        train_set, val_set, batch_size, seed, num_workers
    )
    meta = {"vocab_size": CIFAR_VOCAB, "n_classes": CIFAR_CLASSES, "max_len": CIFAR_SEQ_LEN}
    return train_loader, val_loader, meta


# ---- The registry ----

TASK_REGISTRY = {
    "listops": load_listops,
    "cifar": load_cifar,
}


def build_dataloaders(task, data_dir, max_len, batch_size, seed, num_workers=2):
    """Routes parameters to the specified task loader.
    """
    if task not in TASK_REGISTRY:
        raise KeyError(
            f"unknown task {task!r}; registered: {sorted(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[task](data_dir, max_len, batch_size, seed, num_workers)