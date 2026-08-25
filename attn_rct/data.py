"""ListOps data loading pipeline.

Ensures identical tokenisation, batch composition, and ordering across all experimental arms.
Sequences are dynamically padded to the batch maximum to save compute, and over-length 
sequences are filtered rather than truncated to preserve expression validity.
"""

from __future__ import annotations

import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def read_tsv(path: Path) -> list[tuple[str, int]]:
    """Reads a single LRA split TSV file, returning a list of (source, target) pairs.

    Raises:
        ValueError: If the file is not in the official Source/Target schema.
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


def build_vocab(rows: list[tuple[str, int]]) -> dict[str, int]:
    """Builds a deterministic vocabulary from the training split."""
    tokens = set()
    for source, _ in rows:
        tokens.update(source.split())
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in sorted(tokens):
        vocab[token] = len(vocab)
    return vocab


class ListOpsDataset(Dataset):
    """Dataset for tokenised ListOps examples. Filters out sequences exceeding max_len."""

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


def collate(batch):
    """Pads each batch to match the length of its longest sequence.

    Returns the padded token IDs, an attention mask (1 for real tokens), and target labels.
    """
    sequences, targets = zip(*batch)
    longest = max(len(s) for s in sequences)
    padded = torch.zeros(len(sequences), longest, dtype=torch.long)
    mask = torch.zeros(len(sequences), longest, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1
    return padded, mask, torch.tensor(targets, dtype=torch.long)


def build_dataloaders(data_dir, max_len, batch_size, seed, num_workers=2):
    """Builds the train and validation dataloaders.

    Uses a seeded generator to ensure identical batch ordering across different experimental runs.

    Raises:
        FileNotFoundError: If either the train or validation split is missing.
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
    vocab = build_vocab(train_rows)

    train_set = ListOpsDataset(train_rows, vocab, max_len)
    val_set = ListOpsDataset(val_rows, vocab, max_len)

    if train_set.n_filtered or val_set.n_filtered:
        kept = len(train_set) / max(len(train_rows), 1)
        print(f"NOTE: filtered {train_set.n_filtered} train and {val_set.n_filtered} val "
              f"sequences longer than max_len={max_len}. {kept:.0%} of train kept.")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=generator,
        collate_fn=collate, num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers,
    )
    return train_loader, val_loader, vocab