"""Tests the Text loader and gradient accumulation.

Since Text's 4000-token sequences won't fit in memory at a batch size of 16, I'm using 
gradient accumulation. To keep the RCT fair, accumulation must not change the actual optimization.
These tests ensure:
  1. Equivalence: 4 micro-batches of 4 give the exact same gradient as 1 batch of 16.
  2. Inertness: With accumulation off, the backward pass is bit-identical to my previous runs.
  
The loader tests also verify that padding trims correctly and the validation split is deterministic.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from attn_rct.data import load_text, _load_text_split
from attn_rct.model import TransformerClassifier
from attn_rct.train import resolve_accumulation


# --------------------------------------------------------------------------------------
# resolve_accumulation
# --------------------------------------------------------------------------------------

def test_no_accumulation_by_default():
    """Rows without the field (like my earlier runs) stay untouched."""
    assert resolve_accumulation(SimpleNamespace(batch_size=16)) == (16, 1)


def test_accumulation_splits_the_batch():
    assert resolve_accumulation(SimpleNamespace(batch_size=16, grad_accum_steps=4)) == (4, 4)


def test_indivisible_batch_is_refused():
    """A remainder would silently alter the effective batch size, so fail early."""
    with pytest.raises(ValueError, match="not divisible"):
        resolve_accumulation(SimpleNamespace(batch_size=16, grad_accum_steps=3))


# --------------------------------------------------------------------------------------
# Gradient equivalence
# --------------------------------------------------------------------------------------

def tiny_model():
    cfg = SimpleNamespace(
        d_model=32, n_heads=4, depth=2, d_ff=64, dropout=0.0, attn_dropout=0.0,
        max_len=24, attention="vanilla", variant="vanilla", seed=0,
        linformer_k=8, linformer_sharing="layerwise", sparse_pattern="fixed",
        sparse_stride=4, compute_dtype="fp32", require_flash=False,
    )
    torch.manual_seed(0)
    return TransformerClassifier(cfg, vocab_size=50, n_classes=3)


def grads(model):
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


def batch(n=16, length=24, seed=1):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, 50, (n, length), generator=g)
    mask = torch.ones(n, length, dtype=torch.long)
    targets = torch.randint(0, 3, (n,), generator=g)
    return tokens, mask, targets


def test_accumulated_gradient_equals_full_batch_gradient():
    """Four micro-batches of 4, each loss divided by 4, equal one batch of 16."""
    tokens, mask, targets = batch()
    criterion = nn.CrossEntropyLoss()

    full = tiny_model()
    criterion(full(tokens, mask), targets).backward()

    accum = tiny_model()
    for chunk in range(4):
        sl = slice(4 * chunk, 4 * chunk + 4)
        (criterion(accum(tokens[sl], mask[sl]), targets[sl]) / 4).backward()

    for a, b in zip(grads(full), grads(accum)):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_accumulation_off_is_bit_identical():
    """Dividing by an accumulation count of 1 shouldn't change a single bit."""
    tokens, mask, targets = batch()
    criterion = nn.CrossEntropyLoss()

    old = tiny_model()
    criterion(old(tokens, mask), targets).backward()

    new = tiny_model()
    (criterion(new(tokens, mask), targets) / 1).backward()

    for a, b in zip(grads(old), grads(new)):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------------------
# Text loader
# --------------------------------------------------------------------------------------

def write_pickle(path, rows):
    records = []
    for ids, label in rows:
        padded = np.zeros(4096, dtype=np.int64)
        padded[:len(ids)] = ids
        records.append({"input_ids_0": padded, "label": label})
    with open(path, "wb") as handle:
        pickle.dump(records, handle)


def test_reviews_are_trimmed_to_real_length(tmp_path):
    path = tmp_path / "lra-text.train.pickle"
    write_pickle(path, [([5, 6, 7], 1), ([9] * 10, 0)])
    sequences, labels, n_trunc = _load_text_split(path, 4000)
    assert [len(s) for s in sequences] == [3, 10]
    assert list(sequences[0]) == [5, 6, 7]
    assert list(labels) == [1, 0]
    assert n_trunc == 0


def test_long_reviews_are_truncated_not_dropped(tmp_path):
    """Cutting a review's tail shouldn't change its sentiment label, so keep it."""
    path = tmp_path / "lra-text.train.pickle"
    write_pickle(path, [([3] * 50, 1)])
    sequences, _labels, n_trunc = _load_text_split(path, 20)
    assert len(sequences[0]) == 20
    assert n_trunc == 1


def test_a_zero_inside_a_review_is_refused(tmp_path):
    """I assume padding is purely trailing. This ensures I don't get silent errors if a zero appears mid-review."""
    path = tmp_path / "lra-text.train.pickle"
    with open(path, "wb") as handle:
        ids = np.zeros(4096, dtype=np.int64)
        ids[:5] = [4, 4, 0, 4, 4]
        pickle.dump([{"input_ids_0": ids, "label": 1}], handle)
    with pytest.raises(ValueError, match="zero token inside a review"):
        _load_text_split(path, 4000)


def test_validation_split_is_carved_from_train_and_deterministic(tmp_path):
    """Ensures the 10% holdout split is identical every time and never touches the test file."""
    rows = [([i % 200 + 1] * (5 + i % 7), i % 2) for i in range(100)]
    write_pickle(tmp_path / "lra-text.train.pickle", rows)
    # No dev or test file exists: the loader must not need them.
    t1, v1, meta = load_text(tmp_path, 4000, batch_size=4, seed=0, num_workers=0)
    t2, v2, _ = load_text(tmp_path, 4000, batch_size=4, seed=0, num_workers=0)
    assert len(v1.dataset) == 10 and len(t1.dataset) == 90
    a = [s.tolist() for s, _ in v1.dataset]
    b = [s.tolist() for s, _ in v2.dataset]
    assert a == b
    assert meta == {"vocab_size": 256, "n_classes": 2, "max_len": 4000}


def test_batches_pad_to_their_longest_review(tmp_path):
    rows = [([7] * n, n % 2) for n in (3, 9, 5, 4) * 5]
    write_pickle(tmp_path / "lra-text.train.pickle", rows)
    train, _val, _meta = load_text(tmp_path, 4000, batch_size=4, seed=0, num_workers=0)
    tokens, mask, _targets = next(iter(train))
    assert tokens.shape[1] == int(mask.sum(dim=1).max())
    assert (tokens[mask == 0] == 0).all()