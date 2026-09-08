"""Unit tests for the data collection and integrity-checking stage.

Ensures that the collection logic strictly rejects incomplete, corrupted, or 
incompatible experimental data matrices before statistical analysis begins.
"""

import json
from pathlib import Path

import pytest

from attn_rct.analysis import collect, synth


@pytest.fixture
def full_grid(tmp_path):
    """Generates a complete synthetic results grid for testing."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    return tmp_path


def corrupt(directory, run_id, mutate):
    """Modifies a specific JSON result file to simulate data corruption."""
    path = Path(directory) / f"{run_id}.json"
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))


def test_full_grid_is_complete(full_grid):
    """Verifies that a perfectly complete grid loads all rows and designs without errors."""
    df, report = collect.load_results(full_grid, phase="full")
    assert report.n_rows == 120
    assert sorted(report.complete_designs) == [('listops', i) for i in range(8)]
    assert not report.incomplete_designs
    assert not report.integrity_failures
    keep = collect.require_complete(df, report)
    assert len(keep) == 120


def test_phase_filter_excludes_other_phases(full_grid):
    """Ensures that runs from other phases (e.g., 'smoke') are ignored to prevent contamination."""
    corrupt(full_grid, "d000_flash_s0", lambda p: p.update(phase="smoke"))
    df, report = collect.load_results(full_grid, phase="full")
    
    assert ("listops", 0) in report.incomplete_designs
    assert "flash_s0" in report.incomplete_designs[("listops", 0)]


def test_incomplete_design_is_dropped_not_analysed(full_grid):
    """Checks that a single missing cell correctly invalidates its entire design block."""
    (Path(full_grid) / "d004_vanilla_s2.json").unlink()
    df, report = collect.load_results(full_grid, phase="full")
    
    assert ("listops", 4) in report.incomplete_designs
    keep = collect.require_complete(df, report, drop_incomplete=True)
    assert 4 not in keep["design_id"].unique()
    assert sorted(keep["design_id"].unique()) == [0, 1, 2, 3, 5, 6, 7]


def test_incomplete_is_hard_error_when_not_dropping(full_grid):
    """Ensures the collector raises a fatal error on incomplete blocks when strict mode is enforced."""
    (Path(full_grid) / "d004_sparse_s1.json").unlink()
    df, report = collect.load_results(full_grid, phase="full")
    
    with pytest.raises(ValueError, match="incomplete blocks"):
        collect.require_complete(df, report, drop_incomplete=False)


def test_broken_pairing_is_caught(full_grid):
    """Verifies that varying architectural configurations within the same design block trigger a failure."""
    corrupt(full_grid, "d001_vanilla_s0", lambda p: p["config"].update(depth=99))
    df, report = collect.load_results(full_grid, phase="full")
    
    assert any("pairing is broken" in f for f in report.integrity_failures)
    with pytest.raises(ValueError, match="integrity checks failed"):
        collect.require_complete(df, report)


def test_unverified_flash_backend_is_caught(full_grid):
    """Ensures FlashAttention runs are flagged if they fallback to an unverified backend."""
    corrupt(full_grid, "d000_flash_s0", lambda p: p.update(backend="sdpa fallback NOT fa2"))
    df, report = collect.load_results(full_grid, phase="full")
    
    assert any("backend not verified" in f for f in report.integrity_failures)


def test_nonlinformer_param_drift_is_caught(full_grid):
    """Verifies that parameter count inconsistencies across non-Linformer models trigger an error."""
    corrupt(full_grid, "d003_vanilla_s0", lambda p: p.update(n_params=p["n_params"] + 1000))
    df, report = collect.load_results(full_grid, phase="full")
    
    assert any("disagree on n_params" in f for f in report.integrity_failures)


def test_linformer_overhead_wrong_is_caught(full_grid):
    """Checks that an incorrect parameter overhead for Linformer models is flagged."""
    for seed in (0, 1, 2):
        corrupt(full_grid, f"d002_linformer_s{seed}", lambda p: p.update(n_params=p["n_params"] + 10_000))
    df, report = collect.load_results(full_grid, phase="full")
    
    assert any("overhead" in f for f in report.integrity_failures)


def test_linformer_disagrees_across_seeds_is_caught(full_grid):
    """Ensures that parameter count drift across random seeds for Linformer triggers an error."""
    corrupt(full_grid, "d002_linformer_s0", lambda p: p.update(n_params=p["n_params"] + 1))
    df, report = collect.load_results(full_grid, phase="full")
    
    assert any("disagree on n_params" in f for f in report.integrity_failures)


def test_partial_training_is_warned(full_grid):
    """Verifies that prematurely terminated runs log a warning instead of failing silently."""
    corrupt(full_grid, "d000_linear_s0", lambda p: p.update(epochs_trained_this_session=5))
    df, report = collect.load_results(full_grid, phase="full")
    
    assert any("partial run" in w for w in report.warnings)


def test_null_effect_still_loads_cleanly(tmp_path):
    """Ensures grids with no statistical effect size still load and process correctly."""
    synth.generate(tmp_path, effect=0.0, seed=3)
    df, report = collect.load_results(tmp_path, phase="full")
    
    assert report.n_rows == 120
    assert not report.integrity_failures


def test_empty_directory_is_handled(tmp_path):
    """Checks that pointing the collector to an empty directory logs a warning rather than crashing."""
    df, report = collect.load_results(tmp_path, phase="full")
    
    assert df.empty
    assert any("no rows" in w for w in report.warnings)
    with pytest.raises(ValueError):
        collect.require_complete(df, report)


# ---- Multi-tasking ----

def _stamp_task(directory, task, prefix, drop=()):
    """Renames a synthetic grid's files with a task prefix and injects the task field into the JSON."""
    for path in list(Path(directory).glob("d[0-9]*.json")):
        payload = json.loads(path.read_text())
        payload["task"] = task
        payload["config"]["task"] = task
        name = f"{prefix}{path.name}"
        path.unlink()
        if name in drop:
            continue
        (Path(directory) / name).write_text(json.dumps(payload))


def test_two_tasks_are_distinct_blocks(tmp_path):
    """Ensures identical design indices from different tasks are treated as completely distinct blocks."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    _stamp_task(tmp_path, "listops", "listops_")
    synth.generate(tmp_path, effect=1.0, seed=1)
    _stamp_task(tmp_path, "cifar", "cifar_")

    df, report = collect.load_results(tmp_path, phase="full")
    
    assert sorted(df["task"].unique()) == ["cifar", "listops"]
    assert len(report.complete_designs) == 16
    assert ("listops", 4) in report.complete_designs
    assert ("cifar", 4) in report.complete_designs


def test_incomplete_block_in_one_task_does_not_drop_the_other(tmp_path):
    """Verifies that an incomplete block in one task does not invalidate the corresponding block in another."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    _stamp_task(tmp_path, "listops", "listops_")
    synth.generate(tmp_path, effect=1.0, seed=1)
    _stamp_task(tmp_path, "cifar", "cifar_", drop=("cifar_d004_sparse_s1.json",))

    df, report = collect.load_results(tmp_path, phase="full")
    
    assert ("cifar", 4) in report.incomplete_designs
    assert ("listops", 4) in report.complete_designs
    
    keep = collect.require_complete(df, report)
    kept_blocks = set(zip(keep["task"], keep["design_id"]))
    
    assert ("listops", 4) in kept_blocks
    assert ("cifar", 4) not in kept_blocks


def test_task_defaults_to_listops_for_old_results(tmp_path):
    """Ensures backwards compatibility by defaulting older result files to the 'listops' task."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    for path in Path(tmp_path).glob("d[0-9]*.json"):
        payload = json.loads(path.read_text())
        payload.pop("task", None)
        payload["config"].pop("task", None)
        path.write_text(json.dumps(payload))
        
    df, _ = collect.load_results(tmp_path, phase="full")
    
    assert (df["task"] == "listops").all()