"""Task data pipelines behind a single registry.

Every task -- ListOps, CIFAR-10, and whatever follows -- has a different raw format but
must present the model with the same thing: a padded batch of token ids, a padding mask,
and integer labels, produced identically across every attention arm. The registry keeps
that contract in one place so the training loop and the model never learn which task they
are running, and so adding a task is a matter of writing one loader rather than branching
through the pipeline.

The unit of the contract is build_dataloaders(task, ...), which returns
(train_loader, val_loader, meta), where meta carries vocab_size, n_classes and the
task's fixed max_len. Batches are (token_ids, mask, targets), the same tuple every arm's
forward pass already expects.
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
    """Pad a batch of (ids, target) to the longest sequence in the batch.

    Returns:
        (padded_ids, mask, targets) with mask 1 for real tokens, 0 for padding.
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
    """Build the train/val DataLoaders with seeded, reproducible batch ordering."""
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


# ---- ListOps (unchanged behaviour, now behind the registry) ----

def read_tsv(path: Path) -> list[tuple[str, int]]:
    """Read one LRA split TSV, returning (source, target) pairs.

    Raises:
        ValueError: if the file is not in the official Source/Target schema.
    """
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "Source" not in reader.fieldnames or "Target" not in reader.fieldnames:
            raise ValueError(
                f"{path} has columns {reader.fieldnames}; expected Source/Target. "
                "Is this the official LRA schema?"
            )
        for row in reader:
            rows.append((row["Source"], int(row["Target"])))
    return rows


def build_listops_vocab(rows) -> dict[str, int]:
    """Deterministic vocabulary from the ListOps training split."""
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
    """ListOps loader. See build_dataloaders for the returned contract.

    Raises:
        FileNotFoundError: if a split is missing.
    """
    data_dir = Path(data_dir)
    files = {split: data_dir / f"basic_{split}.tsv" for split in ("train", "val")}
    for split, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}. Expected the LRA layout basic_train/val/test.tsv."
            )

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


# ---- CIFAR-10 as LRA's Image task: greyscale, flattened to 1024, 8-bit pixel tokens ----

CIFAR_SEQ_LEN = 1024        # 32 x 32
CIFAR_VOCAB = 256           # 8-bit greyscale intensities
CIFAR_CLASSES = 10
# ITU-R BT.601 luma weights, the standard RGB->grey conversion LRA's pipeline uses.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _load_cifar_batch(path: Path):
    """Read one CIFAR-10 pickle, returning (data uint8 [N,3072], labels list)."""
    with open(path, "rb") as handle:
        entry = pickle.load(handle, encoding="bytes")
    return entry[b"data"], entry[b"labels"]


def cifar_to_grey_tokens(data: np.ndarray) -> np.ndarray:
    """Convert raw CIFAR rows to length-1024 sequences of 8-bit greyscale tokens.

    Args:
        data: (N, 3072) uint8, CIFAR's channel-major R,G,B layout.

    Returns:
        (N, 1024) uint8, row-major greyscale pixel intensities in [0, 255].
    """
    n = data.shape[0]
    rgb = data.reshape(n, 3, CIFAR_SEQ_LEN)            # (N, {R,G,B}, 1024)
    grey = np.tensordot(_LUMA, rgb.astype(np.float64), axes=([0], [1]))  # (N, 1024)
    return np.clip(np.round(grey), 0, 255).astype(np.uint8)


class CifarDataset(Dataset):
    """Greyscale-flattened CIFAR-10 as fixed-length token sequences.

    Every example is exactly CIFAR_SEQ_LEN tokens, so nothing is ever filtered or
    truncated -- the length-cap problem that limits ListOps does not arise here.
    """

    def __init__(self, tokens: np.ndarray, labels):
        self.tokens = tokens
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ids = torch.from_numpy(self.tokens[index].astype(np.int64))
        return ids, int(self.labels[index])


def load_cifar(data_dir, max_len, batch_size, seed, num_workers=2, val_fraction=0.1):
    """CIFAR-10 Image-task loader.

    The five data_batch files are the training pool; a deterministic slice becomes the
    validation split, following LRA's practice of holding validation out of the training
    images rather than using the test set for model selection. The test_batch is left for
    a final held-out evaluation and is not touched here.

    Args:
        data_dir: directory containing cifar-10-batches-py/.
        max_len: must be CIFAR_SEQ_LEN; present for interface symmetry and checked.
        batch_size, seed, num_workers: as build_dataloaders.
        val_fraction: fraction of the 50k training images held out for validation.

    Raises:
        FileNotFoundError: if the extracted batches are absent.
        ValueError: if max_len does not match the task's fixed sequence length.
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

    # Deterministic train/val split. The permutation is seeded so the split is identical
    # across every arm and seed of a run, which the paired design requires.
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


# ---- Pathfinder (LRA), the pre-processed pickle form ----
#
# Pathfinder32 asks whether two dots in a 32x32 image are joined by a dashed path. It is
# the LRA task where approximate attention is reported to struggle most, which makes it
# the sharpest test of whether the accuracy ranking seen on ListOps and CIFAR holds.
#
# The pre-processed pickles are a list of {"input_ids_0": int array of PATHFINDER_SEQ_LEN,
# "label": 0/1}. The image is already flattened to a 1024-token sequence of 8-bit pixel
# intensities, so the loader only has to stack and wrap it -- no transform, no filtering,
# no length cap. Unlike CIFAR, Pathfinder ships its own train/dev/test splits, so the
# loader uses .train for training and .dev for validation rather than carving a slice.

PATHFINDER_SEQ_LEN = 1024        # 32 x 32
PATHFINDER_VOCAB = 256           # 8-bit pixel intensities
PATHFINDER_CLASSES = 2           # connected / not connected
# The standard LRA Pathfinder32 config, and the stem of the pickle filenames.
PATHFINDER_STEM = "lra-pathfinder32-curv_contour_length_14"


def _load_pathfinder_split(path: Path):
    """Read one Pathfinder pickle, returning (tokens uint8 [N, L], labels int [N]).

    Raises:
        ValueError: if a record's sequence length is not PATHFINDER_SEQ_LEN, which would
            mean the wrong resolution's pickle was supplied.
    """
    with open(path, "rb") as handle:
        records = pickle.load(handle)
    tokens = np.stack([np.asarray(r["input_ids_0"], dtype=np.int64) for r in records])
    labels = np.asarray([int(r["label"]) for r in records], dtype=np.int64)
    if tokens.shape[1] != PATHFINDER_SEQ_LEN:
        raise ValueError(
            f"{path} has sequence length {tokens.shape[1]}, expected "
            f"{PATHFINDER_SEQ_LEN}; is this the pathfinder32 pickle?"
        )
    return tokens, labels


class PathfinderDataset(Dataset):
    """Pre-tokenised Pathfinder images as fixed-length sequences.

    Every example is exactly PATHFINDER_SEQ_LEN tokens, so nothing is filtered or padded
    and the batch mask is all ones, exactly as for CIFAR.
    """

    def __init__(self, tokens: np.ndarray, labels: np.ndarray):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.tokens[index]), int(self.labels[index])


def load_pathfinder(data_dir, max_len, batch_size, seed, num_workers=2):
    """Pathfinder32 loader, using the dataset's own train and dev splits.

    Args:
        data_dir: directory containing the {stem}.train.pickle and {stem}.dev.pickle files.
        max_len: must be PATHFINDER_SEQ_LEN; checked for interface symmetry.
        batch_size, seed, num_workers: as build_dataloaders.

    Raises:
        FileNotFoundError: if a required split pickle is absent.
        ValueError: if max_len does not match the task's fixed sequence length.
    """
    if max_len != PATHFINDER_SEQ_LEN:
        raise ValueError(
            f"Pathfinder32 is a fixed-length task at {PATHFINDER_SEQ_LEN} tokens; "
            f"max_len={max_len} does not match. Set max_len={PATHFINDER_SEQ_LEN}."
        )

    directory = Path(data_dir)
    files = {split: directory / f"{PATHFINDER_STEM}.{split}.pickle"
             for split in ("train", "dev")}
    for split, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}. Download {PATHFINDER_STEM}.{split}.pickle into {directory}."
            )

    train_tokens, train_labels = _load_pathfinder_split(files["train"])
    val_tokens, val_labels = _load_pathfinder_split(files["dev"])

    train_set = PathfinderDataset(train_tokens, train_labels)
    val_set = PathfinderDataset(val_tokens, val_labels)

    train_loader, val_loader = _make_loaders(
        train_set, val_set, batch_size, seed, num_workers
    )
    meta = {"vocab_size": PATHFINDER_VOCAB, "n_classes": PATHFINDER_CLASSES,
            "max_len": PATHFINDER_SEQ_LEN}
    return train_loader, val_loader, meta


# ---- The registry ----

TASK_REGISTRY = {
    "listops": load_listops,
    "cifar": load_cifar,
    "pathfinder": load_pathfinder,
}


def build_dataloaders(task, data_dir, max_len, batch_size, seed, num_workers=2):
    """Dispatch to a task loader, returning a uniform (train, val, meta) tuple.

    Args:
        task: a key in TASK_REGISTRY.
        data_dir: the task's data directory.
        max_len: the task's sequence length (fixed tasks validate it).
        batch_size, seed, num_workers: standard DataLoader controls.

    Returns:
        (train_loader, val_loader, meta) where meta has vocab_size, n_classes, max_len.
        Batches are (token_ids, mask, targets), identical in shape across tasks and arms.

    Raises:
        KeyError: if the task is not registered.
    """
    if task not in TASK_REGISTRY:
        raise KeyError(
            f"unknown task {task!r}; registered: {sorted(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[task](data_dir, max_len, batch_size, seed, num_workers)
