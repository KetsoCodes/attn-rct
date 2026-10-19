"""Task data pipelines behind a single registry.

Every task has a different raw format but must yield the same structure: a padded batch 
of token ids, a padding mask, and integer labels. The registry centralizes this so 
the training loop remains task-agnostic. 
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
    """Pads a batch of (ids, target) to the longest sequence in the batch."""
    sequences, targets = zip(*batch)
    longest = max(len(s) for s in sequences)
    padded = torch.zeros(len(sequences), longest, dtype=torch.long)
    mask = torch.zeros(len(sequences), longest, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1
    return padded, mask, torch.tensor(targets, dtype=torch.long)


def _make_loaders(train_set, val_set, batch_size, seed, num_workers):
    """Builds the train/val DataLoaders with reproducible batch ordering."""
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


# ---- ListOps ----

def read_tsv(path: Path) -> list[tuple[str, int]]:
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "Source" not in reader.fieldnames or "Target" not in reader.fieldnames:
            raise ValueError(f"{path} missing Source/Target columns. Not official LRA schema?")
        for row in reader:
            rows.append((row["Source"], int(row["Target"])))
    return rows


def build_listops_vocab(rows) -> dict[str, int]:
    tokens = set()
    for source, _ in rows:
        tokens.update(source.split())
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in sorted(tokens):
        vocab[token] = len(vocab)
    return vocab


class ListOpsDataset(Dataset):
    """Tokenised ListOps examples, filtering sequences longer than max_len."""

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
    data_dir = Path(data_dir)
    files = {split: data_dir / f"basic_{split}.tsv" for split in ("train", "val")}
    for split, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}.")

    train_rows = read_tsv(files["train"])
    val_rows = read_tsv(files["val"])
    vocab = build_listops_vocab(train_rows)

    train_set = ListOpsDataset(train_rows, vocab, max_len)
    val_set = ListOpsDataset(val_rows, vocab, max_len)

    if train_set.n_filtered or val_set.n_filtered:
        kept = len(train_set) / max(len(train_rows), 1)
        print(f"NOTE: filtered {train_set.n_filtered} train and {val_set.n_filtered} val "
              f"sequences longer than {max_len}. {kept:.0%} of train kept.")

    train_loader, val_loader = _make_loaders(
        train_set, val_set, batch_size, seed, num_workers
    )
    meta = {"vocab_size": len(vocab), "n_classes": 10, "max_len": max_len}
    return train_loader, val_loader, meta


# ---- CIFAR-10 (LRA Image Task) ----

CIFAR_SEQ_LEN = 1024 
CIFAR_VOCAB = 256    
CIFAR_CLASSES = 10
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)  # RGB to greyscale weights


def _load_cifar_batch(path: Path):
    with open(path, "rb") as handle:
        entry = pickle.load(handle, encoding="bytes")
    return entry[b"data"], entry[b"labels"]


def cifar_to_grey_tokens(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    rgb = data.reshape(n, 3, CIFAR_SEQ_LEN)
    grey = np.tensordot(_LUMA, rgb.astype(np.float64), axes=([0], [1]))
    return np.clip(np.round(grey), 0, 255).astype(np.uint8)


class CifarDataset(Dataset):
    """Greyscale-flattened CIFAR-10 sequences (fixed at 1024)."""

    def __init__(self, tokens: np.ndarray, labels):
        self.tokens = tokens
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ids = torch.from_numpy(self.tokens[index].astype(np.int64))
        return ids, int(self.labels[index])


def load_cifar(data_dir, max_len, batch_size, seed, num_workers=2, val_fraction=0.1):
    """Loads CIFAR-10. Holds out a deterministic validation slice from the training set."""
    if max_len != CIFAR_SEQ_LEN:
        raise ValueError(f"CIFAR-10 requires max_len={CIFAR_SEQ_LEN}.")

    root = Path(data_dir) / "cifar-10-batches-py"
    if not root.exists():
        raise FileNotFoundError(f"Missing {root}. Extract cifar-10-python.tar.gz there.")

    data_parts, label_parts = [], []
    for i in range(1, 6):
        part = root / f"data_batch_{i}"
        if not part.exists():
            raise FileNotFoundError(f"Missing {part}")
        data, labels = _load_cifar_batch(part)
        data_parts.append(data)
        label_parts.extend(labels)

    data = np.concatenate(data_parts, axis=0)
    tokens = cifar_to_grey_tokens(data)
    labels = np.array(label_parts)

    # Seeded split ensures identical train/val sets across arms
    rng = np.random.default_rng(12345)
    order = rng.permutation(len(labels))
    n_val = int(len(labels) * val_fraction)
    val_idx, train_idx = order[:n_val], order[n_val:]

    train_set = CifarDataset(tokens[train_idx], labels[train_idx])
    val_set = CifarDataset(tokens[val_idx], labels[val_idx])

    train_loader, val_loader = _make_loaders(train_set, val_set, batch_size, seed, num_workers)
    meta = {"vocab_size": CIFAR_VOCAB, "n_classes": CIFAR_CLASSES, "max_len": CIFAR_SEQ_LEN}
    return train_loader, val_loader, meta


# ---- Pathfinder32 (LRA) ----

PATHFINDER_SEQ_LEN = 1024 
PATHFINDER_VOCAB = 256    
PATHFINDER_CLASSES = 2    
PATHFINDER_STEM = "lra-pathfinder32-curv_contour_length_14"


def _load_pathfinder_split(path: Path):
    with open(path, "rb") as handle:
        records = pickle.load(handle)
    tokens = np.stack([np.asarray(r["input_ids_0"], dtype=np.int64) for r in records])
    labels = np.asarray([int(r["label"]) for r in records], dtype=np.int64)
    if tokens.shape[1] != PATHFINDER_SEQ_LEN:
        raise ValueError(f"{path} sequence length is {tokens.shape[1]}, expected {PATHFINDER_SEQ_LEN}.")
    return tokens, labels


class PathfinderDataset(Dataset):
    """Pre-tokenised Pathfinder images as fixed-length sequences."""

    def __init__(self, tokens: np.ndarray, labels: np.ndarray):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.tokens[index]), int(self.labels[index])


def load_pathfinder(data_dir, max_len, batch_size, seed, num_workers=2):
    if max_len != PATHFINDER_SEQ_LEN:
        raise ValueError(f"Pathfinder requires max_len={PATHFINDER_SEQ_LEN}.")

    directory = Path(data_dir)
    files = {split: directory / f"{PATHFINDER_STEM}.{split}.pickle" for split in ("train", "dev")}
    
    for split, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}.")

    train_tokens, train_labels = _load_pathfinder_split(files["train"])
    val_tokens, val_labels = _load_pathfinder_split(files["dev"])

    train_set = PathfinderDataset(train_tokens, train_labels)
    val_set = PathfinderDataset(val_tokens, val_labels)

    train_loader, val_loader = _make_loaders(train_set, val_set, batch_size, seed, num_workers)
    meta = {"vocab_size": PATHFINDER_VOCAB, "n_classes": PATHFINDER_CLASSES, "max_len": PATHFINDER_SEQ_LEN}
    return train_loader, val_loader, meta


# ---- Text (LRA: IMDb Sentiment) ----

TEXT_MAX_LEN = 4000 
TEXT_VOCAB = 256    
TEXT_CLASSES = 2
TEXT_STEM = "lra-text"


class TextDataset(Dataset):
    """Byte-level reviews trimmed to their real (non-padding) length."""

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.sequences[index]), int(self.labels[index])


def _load_text_split(path: Path, max_len: int):
    """Reads a pickle, trimming each review to its real length (truncates if > max_len)."""
    with open(path, "rb") as handle:
        records = pickle.load(handle)
        
    sequences, labels, n_truncated = [], [], 0
    for record in records:
        ids = np.asarray(record["input_ids_0"], dtype=np.int64)
        real = int(np.count_nonzero(ids))
        
        if real == 0:
            raise ValueError(f"{path.name}: empty review (all padding)")
            
        # Ensure padding is strictly trailing to prevent silent truncation bugs
        if np.count_nonzero(ids[:real]) != real:
            raise ValueError(f"{path.name}: zero token inside a review; padding is not purely trailing.")
            
        if real > max_len:
            n_truncated += 1
            real = max_len
            
        sequences.append(ids[:real].copy())
        labels.append(int(record["label"]))
        
    return sequences, np.asarray(labels, dtype=np.int64), n_truncated


def load_text(data_dir, max_len, batch_size, seed, num_workers=2, val_fraction=0.1):
    """Loads IMDb text task. Carves a deterministic validation slice from the training split."""
    path = Path(data_dir) / f"{TEXT_STEM}.train.pickle"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    sequences, labels, n_truncated = _load_text_split(path, max_len)
    if n_truncated:
        print(f"NOTE: truncated {n_truncated} reviews longer than max_len={max_len}.")

    rng = np.random.default_rng(12345)
    order = rng.permutation(len(labels))
    n_val = int(len(labels) * val_fraction)
    val_idx, train_idx = order[:n_val], order[n_val:]

    train_set = TextDataset([sequences[i] for i in train_idx], labels[train_idx])
    val_set = TextDataset([sequences[i] for i in val_idx], labels[val_idx])

    train_loader, val_loader = _make_loaders(train_set, val_set, batch_size, seed, num_workers)
    meta = {"vocab_size": TEXT_VOCAB, "n_classes": TEXT_CLASSES, "max_len": max_len}
    return train_loader, val_loader, meta


# ---- Task Registry ----

TASK_REGISTRY = {
    "listops": load_listops,
    "cifar": load_cifar,
    "pathfinder": load_pathfinder,
    "text": load_text,
}


def build_dataloaders(task, data_dir, max_len, batch_size, seed, num_workers=2):
    """Dispatches to the correct loader and returns (train_loader, val_loader, meta)."""
    if task not in TASK_REGISTRY:
        raise KeyError(f"Unknown task {task!r}. Registered: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[task](data_dir, max_len, batch_size, seed, num_workers)