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
    assert sorted(report.complete_designs) == [('listops', i) for i in range(8)]
    assert not report.incomplete_designs
    assert not report.integrity_failures
    keep = collect.require_complete(df, report)
    assert len(keep) == 120


def test_phase_filter_excludes_other_phases(full_grid):
    """Rows from another phase are not loaded, so smoke runs never contaminate."""
    corrupt(full_grid, "d000_flash_s0", lambda p: p.update(phase="smoke"))
    df, report = collect.load_results(full_grid, phase="full")
    # d000 flash s0 is now phase=smoke, so d000 is incomplete under phase=full.
    assert ("listops", 0) in report.incomplete_designs
    assert "flash_s0" in report.incomplete_designs[("listops", 0)]


def test_incomplete_design_is_dropped_not_analysed(full_grid):
    """A missing cell drops its whole design; the rest remain analysable."""
    (Path(full_grid) / "d004_vanilla_s2.json").unlink()
    df, report = collect.load_results(full_grid, phase="full")
    assert ("listops", 4) in report.incomplete_designs
    keep = collect.require_complete(df, report, drop_incomplete=True)
    assert 4 not in keep["design_id"].unique()
    assert sorted(keep["design_id"].unique()) == [0, 1, 2, 3, 5, 6, 7]


def test_incomplete_is_hard_error_when_not_dropping(full_grid):
    """With drop_incomplete=False, any hole is fatal."""
    (Path(full_grid) / "d004_sparse_s1.json").unlink()
    df, report = collect.load_results(full_grid, phase="full")
    with pytest.raises(ValueError, match="incomplete blocks"):
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


# ---- multi-task: blocks are (task, design), not design alone ----

def _stamp_task(directory, task, prefix, drop=()):
    """Rename a synthetic grid's files with a task prefix and stamp the task field."""
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
    """The same design index under two tasks is two blocks, not one merged block."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    _stamp_task(tmp_path, "listops", "listops_")
    synth.generate(tmp_path, effect=1.0, seed=1)
    _stamp_task(tmp_path, "cifar", "cifar_")

    df, report = collect.load_results(tmp_path, phase="full")
    assert sorted(df["task"].unique()) == ["cifar", "listops"]
    # 16 complete blocks: 8 designs x 2 tasks, none merged.
    assert len(report.complete_designs) == 16
    assert ("listops", 4) in report.complete_designs
    assert ("cifar", 4) in report.complete_designs


def test_incomplete_block_in_one_task_does_not_drop_the_other(tmp_path):
    """A hole in cifar/d004 must not remove listops/d004, now that they are separate."""
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
    """Results predating the task field are read as listops, their only possible task."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    for path in Path(tmp_path).glob("d[0-9]*.json"):
        payload = json.loads(path.read_text())
        payload.pop("task", None)
        payload["config"].pop("task", None)
        path.write_text(json.dumps(payload))
    df, _ = collect.load_results(tmp_path, phase="full")
    assert (df["task"] == "listops").all()


def test_linformer_overhead_is_task_aware(tmp_path):
    """Linformer overhead is k*max_len, so a task with a different max_len is not a failure.

    CIFAR runs at max_len 1024, giving overhead 256*1024 = 262,144, not ListOps' 512,000.
    The check must compute the expectation from the block, not hardcode one task's value.
    """
    synth.generate(tmp_path, effect=1.0, seed=0)
    # Restamp every file as a cifar block at max_len 1024, and set linformer n_params to
    # the CORRECT cifar overhead so the check should pass, not fire.
    for path in list(Path(tmp_path).glob("d[0-9]*.json")):
        payload = json.loads(path.read_text())
        payload["task"] = "cifar"
        payload["config"]["task"] = "cifar"
        payload["config"]["max_len"] = 1024
        payload["config"]["linformer_k"] = 256
        if payload["variant"] == "linformer":
            # base non-linformer count in the synth generator is design-dependent; recompute
            base = payload["n_params"] - 512_000      # strip the listops overhead the synth added
            payload["n_params"] = base + 256 * 1024    # apply the correct cifar overhead
        (Path(tmp_path) / f"cifar_{path.name}").write_text(json.dumps(payload))
        path.unlink()

    df, report = collect.load_results(tmp_path, phase="full")
    overhead_failures = [f for f in report.integrity_failures if "overhead" in f]
    assert not overhead_failures, f"task-aware overhead check false-fired: {overhead_failures}"


def test_linformer_wrong_overhead_still_caught_per_task(tmp_path):
    """A genuinely wrong overhead is still flagged, with the task's expected value."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    for path in list(Path(tmp_path).glob("d[0-9]*.json")):
        payload = json.loads(path.read_text())
        payload["task"] = "cifar"
        payload["config"]["task"] = "cifar"
        payload["config"]["max_len"] = 1024
        payload["config"]["linformer_k"] = 256
        if payload["variant"] == "linformer":
            payload["n_params"] += 99          # break it
        (Path(tmp_path) / f"cifar_{path.name}").write_text(json.dumps(payload))
        path.unlink()

    df, report = collect.load_results(tmp_path, phase="full")
    assert any("262,144" in f for f in report.integrity_failures)


def retask(src_dir, task, max_len, out_dir=None):
    """Rewrite a synthetic grid as another task's results, named with that task's prefix.

    The grid writes unprefixed `d000_flash_s0.json` for ListOps; real runs write
    `<task>_d000_flash_s0.json`. This mirrors that naming so the collector is tested
    against the filenames the cluster actually produces.
    """
    out_dir = Path(out_dir or src_dir)
    for path in list(Path(src_dir).glob("d[0-9]*.json")):
        payload = json.loads(path.read_text())
        payload["task"] = task
        payload["config"]["task"] = task
        payload["config"]["max_len"] = max_len
        if payload["variant"] == "linformer":
            payload["n_params"] += payload["config"]["linformer_k"] * max_len - 512_000
        (out_dir / f"{task}_{path.name}").write_text(json.dumps(payload))
        path.unlink()


def test_pathfinder_results_are_collected(tmp_path):
    """A task whose results carry its own prefix must still be found.

    This is a regression test. The collector used to look for one glob per task
    ("listops_*", "cifar_*"), so when Pathfinder was added all 120 of its results were
    invisible: the report said two tasks and no check fired, because a file that is never
    opened cannot fail an integrity check. Recognition is now by payload, not filename.
    """
    synth.generate(tmp_path, effect=1.0, seed=0)
    retask(tmp_path, "pathfinder", 1024)

    df, report = collect.load_results(tmp_path, phase="full")
    assert report.n_rows == 120
    assert set(df["task"]) == {"pathfinder"}
    assert sorted(report.complete_designs) == [("pathfinder", i) for i in range(8)]
    assert not report.integrity_failures


def test_an_unregistered_future_task_is_collected_too(tmp_path):
    """Adding a fourth task must not require editing the collector.

    The original bug was a list that had to be kept in lockstep with the task registry.
    A task name the collector has never heard of proves that coupling is gone.
    """
    synth.generate(tmp_path, effect=1.0, seed=0)
    retask(tmp_path, "text", 4000)

    df, report = collect.load_results(tmp_path, phase="full")
    assert report.n_rows == 120
    assert set(df["task"]) == {"text"}


def test_non_result_json_is_ignored_silently(tmp_path):
    """Probe output shares this directory and must neither load nor warn.

    Globbing every *.json is only safe if non-results are rejected on content. They are
    also not errors, so they must not appear as warnings -- a report full of noise about
    files it was never meant to read is a report nobody checks.
    """
    synth.generate(tmp_path, effect=1.0, seed=0)
    (Path(tmp_path) / "memory_probe.json").write_text(
        json.dumps({"peak_memory_mb": 1234, "variant": "vanilla", "batch_size": 32})
    )
    (Path(tmp_path) / "notes.json").write_text(json.dumps({"anything": "at all"}))

    df, report = collect.load_results(tmp_path, phase="full")
    assert report.n_files == 120
    assert report.n_rows == 120
    assert not report.warnings


def test_two_tasks_side_by_side_are_both_collected(tmp_path):
    """Both tasks' blocks appear, keyed on (task, design), with no cross-task merging."""
    listops_dir = Path(tmp_path) / "listops_src"
    synth.generate(listops_dir, effect=1.0, seed=0)
    retask(listops_dir, "pathfinder", 1024, out_dir=tmp_path)
    synth.generate(tmp_path, effect=1.0, seed=1)

    df, report = collect.load_results(tmp_path, phase="full")
    assert report.n_rows == 240
    assert set(df["task"]) == {"listops", "pathfinder"}
    assert len(report.complete_designs) == 16


def test_nonfinite_gradient_steps_are_an_integrity_failure(full_grid):
    """A cell whose updates were zeroed by clipping is not a measurement.

    This is the Pathfinder lr 1e-2 failure mode: clip_grad_norm_ scales by
    max_norm/total_norm, so a non-finite norm gives a coefficient of 1/inf = 0 and the
    optimiser applies nothing. No exception, no NaN loss -- just a model that stops
    learning while the run reports success.
    """
    corrupt(full_grid, "d002_linear_s1", lambda p: p.update(nonfinite_grad_steps=417))
    _df, report = collect.load_results(full_grid, phase="full")
    assert any("non-finite gradient" in f for f in report.integrity_failures)
    assert any("417" in f for f in report.integrity_failures)


def test_zero_nonfinite_steps_is_clean(full_grid):
    """The healthy case must not fire, or the check becomes noise nobody reads."""
    _df, report = collect.load_results(full_grid, phase="full")
    assert not report.integrity_failures
    assert not any("non-finite" in w for w in report.warnings)


def test_missing_counter_is_unknown_not_zero(full_grid):
    """Runs predating the counter must be reported as unknown, not assumed healthy.

    The ListOps and CIFAR results were produced before the counter existed, so whether any
    of their steps were zeroed cannot be established after the fact. Recording that as a
    warning is honest; defaulting it to 0 would be a claim the data does not support.
    """
    for path in Path(full_grid).glob("d[0-9]*.json"):
        payload = json.loads(path.read_text())
        payload.pop("nonfinite_grad_steps", None)
        path.write_text(json.dumps(payload))

    _df, report = collect.load_results(full_grid, phase="full")
    assert any("predate the non-finite-gradient counter" in w for w in report.warnings)
    assert not report.integrity_failures


def test_a_block_with_no_results_is_reported_as_absent(full_grid):
    """A block with zero result files must be named, not silently omitted.

    This is a regression test. Completeness used to be checked by iterating over the
    blocks PRESENT in the table, so a block with no files never entered the table and
    nothing checked it. On the real grid that hid pathfinder/d006 and d007 entirely: the
    report said 18 complete and 4 incomplete and mentioned nothing else, and only the row
    count gave it away. A check that cannot fire on missing data is not a check.
    """
    for path in Path(full_grid).glob("d006_*.json"):
        path.unlink()

    _df, report = collect.load_results(full_grid, phase="full")
    assert ("listops", 6) in report.absent_blocks
    assert ("listops", 6) not in report.incomplete_designs
    assert "ABSENT blocks" in report.summary()
    assert "listops/d006" in report.summary()


def test_absent_and_incomplete_are_distinguished(full_grid):
    """Partly missing and entirely missing are different problems and are reported apart."""
    for path in Path(full_grid).glob("d006_*.json"):
        path.unlink()
    (Path(full_grid) / "d003_sparse_s1.json").unlink()

    _df, report = collect.load_results(full_grid, phase="full")
    assert report.absent_blocks == [("listops", 6)]
    assert list(report.incomplete_designs) == [("listops", 3)]


def test_absent_blocks_are_fatal_when_not_dropping(full_grid):
    """drop_incomplete=False must refuse an absent block as readily as a partial one."""
    for path in Path(full_grid).glob("d006_*.json"):
        path.unlink()

    df, report = collect.load_results(full_grid, phase="full")
    with pytest.raises(ValueError, match="no results at all"):
        collect.require_complete(df, report, drop_incomplete=False)


def test_absence_is_checked_per_task(full_grid, tmp_path):
    """Each task is expected to carry the whole design space, independently.

    A second task missing a block must be flagged even when the first task is complete,
    which is what makes the check useful as tasks are added.
    """
    retask(full_grid, "pathfinder", 1024, out_dir=tmp_path)
    for path in Path(tmp_path).glob("pathfinder_d007_*.json"):
        path.unlink()
    synth.generate(tmp_path, effect=1.0, seed=1)

    _df, report = collect.load_results(tmp_path, phase="full")
    assert ("pathfinder", 7) in report.absent_blocks
    assert not any(task == "listops" for task, _ in report.absent_blocks)


def test_complete_grid_has_no_absent_blocks(full_grid):
    """The healthy case must stay silent, or the field becomes noise."""
    _df, report = collect.load_results(full_grid, phase="full")
    assert report.absent_blocks == []
    assert "ABSENT" not in report.summary()
