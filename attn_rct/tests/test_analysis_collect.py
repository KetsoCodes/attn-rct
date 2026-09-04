"""Tests for the collection and integrity-checking stage.

These lean on the synthetic generator, which writes results in train.py's exact schema.
The point of most of them is that the checks REFUSE bad data: a check that never fires is
worse than no check, because it looks like safety. So each integrity test deliberately
corrupts one thing and asserts the corresponding failure is raised.
"""

import json
from pathlib import Path

import pytest

from attn_rct.analysis import collect, synth


@pytest.fixture
def full_grid(tmp_path):
    """A complete synthetic grid with a planted effect."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    return tmp_path


def corrupt(directory, run_id, mutate):
    """Apply mutate() to one result file's parsed JSON and write it back."""
    path = Path(directory) / f"{run_id}.json"
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))


def test_full_grid_is_complete(full_grid):
    """A clean grid loads 120 rows, 8 complete designs, no failures."""
    df, report = collect.load_results(full_grid, phase="full")
    assert report.n_rows == 120
    assert sorted(report.complete_designs) == list(range(8))
    assert not report.incomplete_designs
    assert not report.integrity_failures
    keep = collect.require_complete(df, report)
    assert len(keep) == 120


def test_phase_filter_excludes_other_phases(full_grid):
    """Rows from another phase are not loaded, so smoke runs never contaminate."""
    corrupt(full_grid, "d000_flash_s0", lambda p: p.update(phase="smoke"))
    df, report = collect.load_results(full_grid, phase="full")
    # d000 flash s0 is now phase=smoke, so d000 is incomplete under phase=full.
    assert 0 in report.incomplete_designs
    assert "flash_s0" in report.incomplete_designs[0]


def test_incomplete_design_is_dropped_not_analysed(full_grid):
    """A missing cell drops its whole design; the rest remain analysable."""
    (Path(full_grid) / "d004_vanilla_s2.json").unlink()
    df, report = collect.load_results(full_grid, phase="full")
    assert 4 in report.incomplete_designs
    keep = collect.require_complete(df, report, drop_incomplete=True)
    assert 4 not in keep["design_id"].unique()
    assert sorted(keep["design_id"].unique()) == [0, 1, 2, 3, 5, 6, 7]


def test_incomplete_is_hard_error_when_not_dropping(full_grid):
    """With drop_incomplete=False, any hole is fatal."""
    (Path(full_grid) / "d004_sparse_s1.json").unlink()
    df, report = collect.load_results(full_grid, phase="full")
    with pytest.raises(ValueError, match="incomplete designs"):
        collect.require_complete(df, report, drop_incomplete=False)


def test_broken_pairing_is_caught(full_grid):
    """A design whose cells disagree on a pairing field is invalid."""
    corrupt(full_grid, "d001_vanilla_s0",
            lambda p: p["config"].update(depth=99))
    df, report = collect.load_results(full_grid, phase="full")
    assert any("pairing is broken" in f for f in report.integrity_failures)
    with pytest.raises(ValueError, match="integrity checks failed"):
        collect.require_complete(df, report)


def test_unverified_flash_backend_is_caught(full_grid):
    """A flash cell that did not take the FA-2 path is flagged."""
    corrupt(full_grid, "d000_flash_s0",
            lambda p: p.update(backend="sdpa fallback NOT fa2"))
    df, report = collect.load_results(full_grid, phase="full")
    assert any("backend not verified" in f for f in report.integrity_failures)


def test_nonlinformer_param_drift_is_caught(full_grid):
    """Non-linformer arms in a design must share a parameter count."""
    corrupt(full_grid, "d003_vanilla_s0",
            lambda p: p.update(n_params=p["n_params"] + 1000))
    df, report = collect.load_results(full_grid, phase="full")
    assert any("disagree on n_params" in f for f in report.integrity_failures)


def test_linformer_overhead_wrong_is_caught(full_grid):
    """If every linformer cell shares a wrong overhead, the gap check catches it."""
    for seed in (0, 1, 2):
        corrupt(full_grid, f"d002_linformer_s{seed}",
                lambda p: p.update(n_params=p["n_params"] + 10_000))
    df, report = collect.load_results(full_grid, phase="full")
    assert any("overhead" in f for f in report.integrity_failures)


def test_linformer_disagrees_across_seeds_is_caught(full_grid):
    """If only one linformer seed drifts, the cross-seed check catches it."""
    corrupt(full_grid, "d002_linformer_s0",
            lambda p: p.update(n_params=p["n_params"] + 1))
    df, report = collect.load_results(full_grid, phase="full")
    assert any("disagree on n_params" in f for f in report.integrity_failures)


def test_partial_training_is_warned(full_grid):
    """A cell that trained fewer epochs than the target is flagged, not silently kept."""
    corrupt(full_grid, "d000_linear_s0",
            lambda p: p.update(epochs_trained_this_session=5))
    df, report = collect.load_results(full_grid, phase="full")
    assert any("partial run" in w for w in report.warnings)


def test_null_effect_still_loads_cleanly(tmp_path):
    """The null case (no true difference) is structurally identical; only values differ."""
    synth.generate(tmp_path, effect=0.0, seed=3)
    df, report = collect.load_results(tmp_path, phase="full")
    assert report.n_rows == 120
    assert not report.integrity_failures


def test_empty_directory_is_handled(tmp_path):
    """No files is a warning and an empty frame, not a crash."""
    df, report = collect.load_results(tmp_path, phase="full")
    assert df.empty
    assert any("no rows" in w for w in report.warnings)
    with pytest.raises(ValueError):
        collect.require_complete(df, report)
