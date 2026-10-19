"""Tests the task registry, CIFAR-10, and Pathfinder loaders.

Ensures every task returns identical structures: (ids, mask, targets) plus a uniform meta. 
Also verifies task-specific transforms like greyscale conversion, fixed lengths, and 
deterministic train/val splits to maintain parity with the published LRA benchmarks.
"""

import pickle

import numpy as np
import pytest
import torch

from attn_rct import data


# ---- fixtures ----

def write_cifar(tmp_path, per_batch=100, seed=0):
    """Creates a format-accurate miniature CIFAR-10 dataset."""
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
    assert {"listops", "cifar"} <= set(data.TASK_REGISTRY)


def test_unknown_task_is_refused(tmp_path):
    with pytest.raises(KeyError, match="unknown task"):
        data.build_dataloaders("mnist", tmp_path, 1024, 8, seed=0, num_workers=0)


# ---- CIFAR greyscale conversion ----

def test_pure_red_greyscale_value():
    """BT.601 luma of pure red is 0.299 * 255 ~ 76."""
    red = np.zeros((1, 3072), dtype=np.uint8)
    red[0, :1024] = 255
    tokens = data.cifar_to_grey_tokens(red)
    assert tokens[0, 0] == round(0.299 * 255)


def test_pure_green_greyscale_value():
    green = np.zeros((1, 3072), dtype=np.uint8)
    green[0, 1024:2048] = 255
    tokens = data.cifar_to_grey_tokens(green)
    assert tokens[0, 0] == round(0.587 * 255)


def test_greyscale_output_is_uint8_length_1024():
    raw = np.random.default_rng(0).integers(0, 256, (7, 3072), dtype=np.uint8)
    tokens = data.cifar_to_grey_tokens(raw)
    assert tokens.shape == (7, 1024)
    assert tokens.dtype == np.uint8
    assert tokens.min() >= 0 and tokens.max() <= 255


def test_white_and_black_map_to_extremes():
    """Ensures white maps to 255 and black to 0."""
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
    """Checks that every CIFAR batch is fixed length with an all-ones mask."""
    parent = write_cifar(tmp_path)
    train, _, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=0, num_workers=0)
    ids, mask, targets = next(iter(train))
    assert ids.shape == (8, 1024)
    assert mask.shape == (8, 1024)
    assert targets.shape == (8,)
    assert bool(mask.all())
    assert ids.dtype == torch.long
    assert 0 <= int(ids.min()) and int(ids.max()) <= 255


def test_cifar_wrong_max_len_is_refused(tmp_path):
    """Fails if max_len doesn't match the fixed sequence length."""
    parent = write_cifar(tmp_path)
    with pytest.raises(ValueError, match="fixed-length task"):
        data.build_dataloaders("cifar", parent, 2000, 8, seed=0, num_workers=0)


def test_cifar_missing_data_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="cifar-10-batches-py"):
        data.build_dataloaders("cifar", tmp_path, 1024, 8, seed=0, num_workers=0)


def test_cifar_val_split_is_ten_percent(tmp_path):
    """Ensures we hold out exactly 10% of the training pool for validation."""
    parent = write_cifar(tmp_path, per_batch=100)
    train, val, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=0, num_workers=0)
    assert len(val.dataset) == 50
    assert len(train.dataset) == 450


def test_cifar_split_is_deterministic_across_seeds(tmp_path):
    """Train/val splits must remain identical across seeds to support the paired design."""
    parent = write_cifar(tmp_path)
    _, val0, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=0, num_workers=0)
    _, val1, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=1, num_workers=0)
    labels0 = np.array(val0.dataset.labels)
    labels1 = np.array(val1.dataset.labels)
    assert np.array_equal(labels0, labels1)
    assert np.array_equal(val0.dataset.tokens, val1.dataset.tokens)


def test_cifar_train_and_val_are_disjoint(tmp_path):
    """Ensures no image leaks between train and validation splits."""
    parent = write_cifar(tmp_path)
    train, val, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=0, num_workers=0)
    train_rows = {row.tobytes() for row in train.dataset.tokens}
    val_rows = {row.tobytes() for row in val.dataset.tokens}
    assert train_rows.isdisjoint(val_rows)


def test_batch_order_depends_on_seed(tmp_path):
    """Different seeds should shuffle the training batches differently."""
    parent = write_cifar(tmp_path)
    train0, _, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=0, num_workers=0)
    train1, _, _ = data.build_dataloaders("cifar", parent, 1024, 8, seed=1, num_workers=0)
    first0 = next(iter(train0))[2]
    first1 = next(iter(train1))[2]
    assert not torch.equal(first0, first1)


# ---- Pathfinder ----

def write_pathfinder(tmp_path, n_train=200, n_dev=60, seed=0):
    """Creates format-accurate miniature Pathfinder train/dev pickles."""
    rng = np.random.default_rng(seed)
    for split, n in [("train", n_train), ("dev", n_dev)]:
        recs = [{"input_ids_0": rng.integers(0, 256, 1024, dtype=np.int32),
                 "label": int(rng.integers(0, 2))} for _ in range(n)]
        path = tmp_path / f"lra-pathfinder32-curv_contour_length_14.{split}.pickle"
        with open(path, "wb") as handle:
            pickle.dump(recs, handle)
    return tmp_path


def test_pathfinder_meta_is_correct(tmp_path):
    write_pathfinder(tmp_path)
    _, _, meta = data.build_dataloaders("pathfinder", tmp_path, 1024, 8, seed=0, num_workers=0)
    assert meta == {"vocab_size": 256, "n_classes": 2, "max_len": 1024}


def test_pathfinder_batch_shape_and_mask(tmp_path):
    """Checks that Pathfinder batches are fixed length with an all-ones mask."""
    write_pathfinder(tmp_path)
    train, _, _ = data.build_dataloaders("pathfinder", tmp_path, 1024, 8, seed=0, num_workers=0)
    ids, mask, targets = next(iter(train))
    assert ids.shape == (8, 1024)
    assert bool(mask.all())
    assert set(targets.tolist()) <= {0, 1}
    assert 0 <= int(ids.min()) and int(ids.max()) <= 255


def test_pathfinder_uses_own_train_dev_splits(tmp_path):
    """Ensures train and dev sizes match the shipped Pathfinder split files."""
    write_pathfinder(tmp_path, n_train=200, n_dev=60)
    train, val, _ = data.build_dataloaders("pathfinder", tmp_path, 1024, 8, seed=0, num_workers=0)
    assert len(train.dataset) == 200
    assert len(val.dataset) == 60


def test_pathfinder_wrong_max_len_is_refused(tmp_path):
    write_pathfinder(tmp_path)
    with pytest.raises(ValueError, match="fixed-length task"):
        data.build_dataloaders("pathfinder", tmp_path, 2000, 8, seed=0, num_workers=0)


def test_pathfinder_missing_split_is_refused(tmp_path):
    """Fails if a split is missing to prevent silently training without validation."""
    rng = np.random.default_rng(0)
    recs = [{"input_ids_0": rng.integers(0, 256, 1024, dtype=np.int32), "label": 0}
            for _ in range(10)]
    with open(tmp_path / "lra-pathfinder32-curv_contour_length_14.train.pickle", "wb") as h:
        pickle.dump(recs, h)
    with pytest.raises(FileNotFoundError, match="dev"):
        data.build_dataloaders("pathfinder", tmp_path, 1024, 8, seed=0, num_workers=0)


def test_pathfinder_wrong_resolution_pickle_is_caught(tmp_path):
    """Rejects pickles if the sequence length doesn't match the expected resolution."""
    rng = np.random.default_rng(0)
    for split in ("train", "dev"):
        recs = [{"input_ids_0": rng.integers(0, 256, 4096, dtype=np.int32), "label": 0}
                for _ in range(10)]
        with open(tmp_path / f"lra-pathfinder32-curv_contour_length_14.{split}.pickle", "wb") as h:
            pickle.dump(recs, h)
    with pytest.raises(ValueError, match="sequence length"):
        data.build_dataloaders("pathfinder", tmp_path, 1024, 8, seed=0, num_workers=0)


def test_registry_lists_the_four_tasks():
    """Adding or dropping a task should be a deliberate, visible change."""
    assert set(data.TASK_REGISTRY) == {"listops", "cifar", "pathfinder", "text"}