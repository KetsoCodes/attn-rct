"""Unit tests for the dataset task registry and CIFAR-10 image loader.

Ensures all datasets return consistently shaped data batches (token IDs, attention masks, 
and targets) and that CIFAR-10 specific transformations (greyscale conversion, fixed 
length, and deterministic validation splits) are applied correctly.
"""

import pickle

import numpy as np
import pytest
import torch

from attn_rct import data


# fixtures ---

def write_cifar(tmp_path, per_batch=100, seed=0):
    """Generates a mock CIFAR-10 dataset directory for testing purposes."""
    root = tmp_path / "cifar-10-batches-py"
    root.mkdir()
    rng = np.random.default_rng(seed)
    for i in range(1, 6):
        payload = {
            b"data": rng.integers(0, 256, size=(per_batch, 3072), dtype=np.uint8),
            b"labels": rng.integers(0, 10, size=per_batch).tolist(),
        }
        with open(root / f"data_batch_{i}", "wb") as handle:
            pickle.dump(payload, handle)
    (root / "batches.meta").write_bytes(b"meta")
    return tmp_path


# ---- registry ----

def test_registry_lists_both_tasks():
    assert set(data.TASK_REGISTRY) == {"listops", "cifar"}


def test_unknown_task_is_refused(tmp_path):
    with pytest.raises(KeyError, match="unknown task"):
        data.build_dataloaders("mnist", tmp_path, 1024, 8, seed=0, num_workers=0)


# ---- CIFAR greyscale conversion ----

def test_pure_red_greyscale_value():
    """Verifies that pure red pixels correctly convert to a greyscale value of ~76."""
    red = np.zeros((1, 3072), dtype=np.uint8)
    red[0, :1024] = 255
    tokens = data.cifar_to_grey_tokens(red)
    assert tokens[0, 0] == round(0.299 * 255)


def test_pure_green_greyscale_value():
    """Verifies that pure green pixels correctly convert to a greyscale value of ~150."""
    green = np.zeros((1, 3072), dtype=np.uint8)
    green[0, 1024:2048] = 255
    tokens = data.cifar_to_grey_tokens(green)
    assert tokens[0, 0] == round(0.587 * 255)


def test_greyscale_output_is_uint8_length_1024():
    """Checks that output sequences are strictly 8-bit integers of length 1024."""
    raw = np.random.default_rng(0).integers(0, 256, (7, 3072), dtype=np.uint8)
    tokens = data.cifar_to_grey_tokens(raw)
    assert tokens.shape == (7, 1024)
    assert tokens.dtype == np.uint8
    assert tokens.min() >= 0 and tokens.max() <= 255


def test_white_and_black_map_to_extremes():
    """Ensures pure white and pure black pixels map accurately to 255 and 0."""
    white = np.full((1, 3072), 255, dtype=np.uint8)
    black = np.zeros((1, 3072), dtype=np.uint8)
    assert data.cifar_to_grey_tokens(white)[0, 0] == 255
    assert data.cifar_to_grey_tokens(black)[0, 0] == 0


# ---- CIFAR loader ----

def test_cifar_meta_is_correct(tmp_path):
    parent = write_cifar(tmp_path)
    _, _, meta = data.build_dataloaders("cifar", parent, 1024, 8, seed=0, num_workers=0)
    assert meta == {"vocab_size": 256, "n_classes": 10, "max_len": 1024}


def test_cifar_batch_shape_and_mask(tmp_path):
    """Checks that CIFAR batches output fixed-length sequences without padding."""
    parent = write_cifar(tmp_path)
    train, _, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=0, num_workers=0)
    ids, mask, targets = next(iter(train))
    assert ids.shape == (8, 1024)
    assert mask.shape == (8, 1024)
    assert targets.shape == (8,)
    assert bool(mask.all()), "CIFAR is fixed-length; the mask must be all ones"
    assert ids.dtype == torch.long
    assert 0 <= int(ids.min()) and int(ids.max()) <= 255


def test_cifar_wrong_max_len_is_refused(tmp_path):
    """Ensures the loader rejects sequence lengths that do not match CIFAR-10's fixed size."""
    parent = write_cifar(tmp_path)
    with pytest.raises(ValueError, match="fixed-length task"):
        data.build_dataloaders("cifar", parent, 2000, 8, seed=0, num_workers=0)


def test_cifar_missing_data_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="cifar-10-batches-py"):
        data.build_dataloaders("cifar", tmp_path, 1024, 8, seed=0, num_workers=0)


def test_cifar_val_split_is_ten_percent(tmp_path):
    """Verifies that 10% of the training data is held out for validation."""
    parent = write_cifar(tmp_path, per_batch=100)
    train, val, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=0, num_workers=0)
    assert len(val.dataset) == 50
    assert len(train.dataset) == 450


def test_cifar_split_is_deterministic_across_seeds(tmp_path):
    """Ensures the train/validation data split remains identical across different random seeds."""
    parent = write_cifar(tmp_path)
    _, val0, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=0, num_workers=0)
    _, val1, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=1, num_workers=0)
    labels0 = np.array(val0.dataset.labels)
    labels1 = np.array(val1.dataset.labels)
    assert np.array_equal(labels0, labels1)
    assert np.array_equal(val0.dataset.tokens, val1.dataset.tokens)


def test_cifar_train_and_val_are_disjoint(tmp_path):
    """Confirms that the training and validation datasets share no identical images."""
    parent = write_cifar(tmp_path)
    train, val, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=0, num_workers=0)
    train_rows = {row.tobytes() for row in train.dataset.tokens}
    val_rows = {row.tobytes() for row in val.dataset.tokens}
    assert train_rows.isdisjoint(val_rows)


def test_batch_order_depends_on_seed(tmp_path):
    """Checks that different random seeds produce different batch shuffling orders."""
    parent = write_cifar(tmp_path)
    train0, _, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=0, num_workers=0)
    train1, _, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=1, num_workers=0)
    first0 = next(iter(train0))[2]
    first1 = next(iter(train1))[2]
    assert not torch.equal(first0, first1)