"""Collect results/*.json into a tidy table, refusing to proceed on a broken matrix.

The Friedman test needs a complete design x variant matrix: a single missing cell drops
the whole design. The dangerous failure is not a crash but silent partial data -- 37 of
40 cells present, the test runs happily, and the reported N is not the N you think. So
this stage's job is as much to refuse as to load.

It returns a long DataFrame, one row per (design, variant, seed), plus a report of what
was excluded and why. Nothing here averages or ranks; that is aggregate.py. The split
matters because every exclusion decision should be visible before any statistic is computed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# What a complete grid looks like, mirrored from make_manifest.py.
VARIANTS = ["vanilla", "flash", "linformer", "linear", "sparse"]
SEEDS = [0, 1, 2]
N_DESIGNS = 8
LINFORMER_OVERHEAD = 512_000

# Fields that must agree across every cell of one design for the pairing to hold. These
# are the design's identity plus the fixed frame; if two cells of the same design differ
# on any of them, they were not the same experiment and the pairing is broken.
PAIRING_FIELDS = ["d_model", "depth", "lr", "batch_size", "max_len", "n_heads"]
# Loaded for the linformer overhead check but not required to be paired.
LINFORMER_K_FIELD = "linformer_k"


@dataclass
class CollectionReport:
    """What was found, what was dropped, and why."""

    n_files: int = 0
    n_rows: int = 0
    complete_designs: list = field(default_factory=list)
    incomplete_designs: dict = field(default_factory=dict)   # design_id -> missing cells
    integrity_failures: list = field(default_factory=list)   # human-readable strings
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"files read           : {self.n_files}",
            f"rows                 : {self.n_rows}",
            f"complete blocks      : {len(self.complete_designs)} "
            f"{[f'{t}/d{d:03d}' for t, d in sorted(self.complete_designs)]}",
        ]
        if self.incomplete_designs:
            lines.append(f"incomplete blocks    : {len(self.incomplete_designs)}")
            for (task, design_id), missing in sorted(self.incomplete_designs.items()):
                lines.append(f"    {task}/d{design_id:03d} missing: {missing}")
        if self.integrity_failures:
            lines.append("integrity failures   :")
            lines.extend(f"    {f}" for f in self.integrity_failures)
        if self.warnings:
            lines.append("warnings             :")
            lines.extend(f"    {w}" for w in self.warnings)
        return "\n".join(lines)


#: A result file is identified by carrying all of these, which together name the cell the
#: row belongs to. Anything missing one of them is some other JSON sharing the directory.
RESULT_SIGNATURE = ["run_id", "phase", "task", "design_id", "variant", "seed"]


def _is_result_file(row: dict) -> bool:
    """Is this parsed payload a training result, as opposed to a probe or stray JSON?

    Judged on whether the payload names the cell it belongs to. `_load_one` uses `.get`
    throughout, so a non-result file yields a row of Nones rather than raising, and
    without this check it would be counted as a file that merely failed the phase filter.
    """
    return all(row.get(field) is not None for field in RESULT_SIGNATURE)


def _load_one(path: Path) -> dict:
    """Read one result file and flatten the fields the analysis needs into a flat row."""
    payload = json.loads(path.read_text())
    config = payload.get("config", {})
    row = {
        "run_id": payload.get("run_id"),
        "phase": payload.get("phase"),
        # Task lives in config for grid runs. Older ListOps results predate task being a
        # top-level field, so fall back to config then to the historical default, which
        # was always listops before the multi-task extension.
        "task": payload.get("task") or config.get("task", "listops"),
        "design_id": payload.get("design_id"),
        "variant": payload.get("variant"),
        "seed": payload.get("seed"),
        "n_params": payload.get("n_params"),
        "backend": payload.get("backend"),
        "best_val_accuracy": payload.get("best_val_accuracy"),
        "final_val_accuracy": payload.get("final_val_accuracy"),
        "mean_train_seconds_per_epoch": payload.get("mean_train_seconds_per_epoch"),
        "peak_memory_mb": payload.get("peak_memory_mb"),
        "epochs_trained_this_session": payload.get("epochs_trained_this_session"),
        # Steps whose gradient norm was non-finite, and which clipping therefore scaled to
        # a zero update. Absent in runs made before the counter existed, which is why the
        # integrity check treats None as "unknown" rather than as zero.
        "nonfinite_grad_steps": payload.get("nonfinite_grad_steps"),
        "target_epochs": config.get("epochs"),
        "_source": path.name,
    }
    for f in PAIRING_FIELDS:
        row[f] = config.get(f)
    row[LINFORMER_K_FIELD] = config.get(LINFORMER_K_FIELD)
    return row


def load_results(results_dir, phase="full") -> tuple[pd.DataFrame, CollectionReport]:
    """Load every result file for a phase into a long DataFrame.

    Args:
        results_dir: directory containing d*.json result files.
        phase: only rows with this phase are kept, so throwaway smoke runs never
            contaminate a reportable table.

    Returns:
        (df, report). df has one row per (design, variant, seed). report records
        completeness and integrity findings; it does NOT raise -- the caller decides
        whether to proceed via require_complete.
    """
    results_dir = Path(results_dir)
    report = CollectionReport()

    rows = []
    # Every JSON in the directory is a candidate, and a result file is recognised by its
    # own contents rather than by its name. An earlier version listed one glob per task
    # ("listops_*", "cifar_*"), which silently dropped all 120 pathfinder_* results when
    # the third task was added: the analysis reported two tasks and looked correct. A
    # name-based filter has to be updated in lockstep with every new task and gives no
    # sign when it was not, so the recognition rule lives in the data instead.
    for path in sorted(results_dir.glob("*.json")):
        try:
            row = _load_one(path)
        except json.JSONDecodeError as err:
            report.warnings.append(f"could not parse {path.name}: {err}")
            continue
        if not _is_result_file(row):
            # Memory probes and other JSON share this directory. Not a result file is
            # not an error, so it is skipped without a warning and without counting.
            continue
        report.n_files += 1
        if row["phase"] != phase:
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    report.n_rows = len(df)
    if df.empty:
        report.warnings.append(f"no rows with phase={phase!r} in {results_dir}")
        return df, report

    _check_completeness(df, report)
    _check_partial_training(df, report)
    _check_nonfinite_gradients(df, report)
    _check_pairing(df, report)
    _check_params(df, report)
    _check_backend(df, report)
    return df, report


def _blocks(df):
    """Yield (task, design_id) block keys present in the table, in stable order."""
    seen = df[["task", "design_id"]].drop_duplicates()
    seen = seen.sort_values(["task", "design_id"])
    for _, r in seen.iterrows():
        yield r["task"], int(r["design_id"])


def _check_completeness(df, report):
    """Record which (task, design) blocks have all 5 variants x 3 seeds."""
    expected = {(v, s) for v in VARIANTS for s in SEEDS}
    for task, design_id in _blocks(df):
        block = df[(df["task"] == task) & (df["design_id"] == design_id)]
        present = set(map(tuple, block[["variant", "seed"]].values))
        missing = expected - present
        key = (task, design_id)
        if missing:
            report.incomplete_designs[key] = sorted(f"{v}_s{s}" for v, s in missing)
        else:
            report.complete_designs.append(key)


def _check_nonfinite_gradients(df, report):
    """Flag cells that spent steps on gradients clipping scaled to a zero update.

    `clip_grad_norm_` scales by max_norm/total_norm, so a non-finite total_norm yields a
    coefficient of 1/inf = 0 and the optimiser applies nothing. Training then reports
    success with no exception and no NaN loss -- the Pathfinder lr 1e-2 probe cells were
    frozen this way for a hundred epochs, and the only visible symptom was a validation
    loss that was bit-identical between epochs.

    A cell with any such steps did not train for part of its budget, so it is an integrity
    failure rather than a warning: its accuracy is not a measurement of its configuration.
    Runs made before the counter existed report None, which is recorded as unknown rather
    than silently read as zero.
    """
    if "nonfinite_grad_steps" not in df.columns:
        return

    unknown = df["nonfinite_grad_steps"].isna().sum()
    if unknown:
        report.warnings.append(
            f"{unknown} cell(s) predate the non-finite-gradient counter; whether any of "
            f"their updates were zeroed by clipping cannot be established retrospectively"
        )

    affected = df[df["nonfinite_grad_steps"].fillna(0) > 0]
    for _, row in affected.iterrows():
        report.integrity_failures.append(
            f"{row['_source']}: {int(row['nonfinite_grad_steps'])} step(s) had a "
            f"non-finite gradient norm, which clipping scaled to a zero update -- this "
            f"cell did not train for part of its budget"
        )


def _check_partial_training(df, report):
    """Flag cells whose timing reflects fewer epochs than intended.

    A requeued run reports mean_train_seconds_per_epoch over only THIS session's epochs,
    so a cell that resumed near the end reports timing over a partial session. That does
    not corrupt accuracy but it makes the efficiency numbers mean different things across
    cells, so it must be visible rather than silently averaged.
    """
    for _, r in df.iterrows():
        trained, target = r["epochs_trained_this_session"], r["target_epochs"]
        if trained is not None and target is not None and trained < target:
            report.warnings.append(
                f"{r['run_id']}: trained {trained} of {target} epochs this session; "
                "efficiency numbers may cover a partial run"
            )


def _check_pairing(df, report):
    """Every cell of one block must agree on the pairing fields, or the block is invalid."""
    for task, design_id in _blocks(df):
        block = df[(df["task"] == task) & (df["design_id"] == design_id)]
        for f in PAIRING_FIELDS:
            distinct = block[f].dropna().unique()
            if len(distinct) > 1:
                report.integrity_failures.append(
                    f"{task} d{design_id:03d}: {f} varies within block "
                    f"({sorted(distinct)}); pairing is broken"
                )


def _check_params(df, report):
    """Within a block, non-linformer arms share a param count; linformer is +512,000.

    Keyed on (task, design) because CIFAR and ListOps have different vocab sizes and so
    different embedding parameter counts -- pooling them would false-alarm on every block.
    """
    for task, design_id in _blocks(df):
        block = df[(df["task"] == task) & (df["design_id"] == design_id)]
        tag = f"{task} d{design_id:03d}"
        others = block[block["variant"] != "linformer"]["n_params"].dropna().unique()
        if len(others) > 1:
            report.integrity_failures.append(
                f"{tag}: non-linformer arms disagree on n_params "
                f"({sorted(others)}); a config drifted"
            )
        linf = block[block["variant"] == "linformer"]["n_params"].dropna().unique()
        if len(linf) > 1:
            report.integrity_failures.append(
                f"{tag}: linformer arms disagree on n_params "
                f"({sorted(int(x) for x in linf)}); a config drifted"
            )
        if len(others) == 1 and len(linf) == 1:
            gap = int(linf[0]) - int(others[0])
            # Linformer's shared projection costs k * max_len parameters under layerwise
            # sharing, so the expected overhead is task-dependent: 512,000 at ListOps'
            # max_len=2000 but 262,144 at CIFAR's 1024. Compute it from the block rather
            # than hardcoding a single task's value.
            k_values = block[LINFORMER_K_FIELD].dropna().unique()
            max_len_values = block["max_len"].dropna().unique()
            if len(k_values) == 1 and len(max_len_values) == 1:
                expected = int(k_values[0]) * int(max_len_values[0])
                if gap != expected:
                    report.integrity_failures.append(
                        f"{tag}: linformer overhead is {gap:,}, "
                        f"expected {expected:,} (k={int(k_values[0])} x "
                        f"max_len={int(max_len_values[0])})"
                    )
            elif gap <= 0:
                report.integrity_failures.append(
                    f"{tag}: linformer overhead is {gap:,}; expected a positive "
                    "k * max_len and could not verify the exact value"
                )


def _check_backend(df, report):
    """Every flash cell must have taken the verified FA-2 path, or its efficiency is meaningless."""
    flash = df[df["variant"] == "flash"]
    for _, r in flash.iterrows():
        backend = r["backend"] or ""
        if "VERIFIED" not in backend:
            report.integrity_failures.append(
                f"{r['run_id']}: flash backend not verified (backend={backend!r}); "
                "it may have run a fallback kernel"
            )


def require_complete(df, report, drop_incomplete=True):
    """Return only analysable designs, raising if integrity failed or nothing is left.

    Args:
        df: the long DataFrame from load_results.
        report: its CollectionReport.
        drop_incomplete: if True, silently drop incomplete designs (they cannot enter the
            Friedman test anyway) and keep the complete ones. If False, any incomplete
            design is a hard error.

    Returns:
        A DataFrame containing only complete designs.

    Raises:
        ValueError: on any integrity failure, on an incomplete design when
            drop_incomplete is False, or when no complete design remains.
    """
    if report.integrity_failures:
        raise ValueError(
            "integrity checks failed; refusing to analyse:\n  "
            + "\n  ".join(report.integrity_failures)
        )
    if df.empty or "design_id" not in df.columns:
        raise ValueError("no results to analyse (empty result set)")
    if report.incomplete_designs and not drop_incomplete:
        raise ValueError(
            "incomplete blocks present and drop_incomplete=False:\n  "
            + "\n  ".join(f"{task}/d{design_id:03d}: {m}"
                          for (task, design_id), m
                          in sorted(report.incomplete_designs.items()))
        )
    complete = set(report.complete_designs)
    mask = df.apply(lambda r: (r["task"], int(r["design_id"])) in complete, axis=1)
    keep = df[mask].copy()
    if keep.empty:
        raise ValueError(
            "no complete (task, design) x variant matrix available; "
            f"complete blocks: {sorted(report.complete_designs)}"
        )
    return keep
