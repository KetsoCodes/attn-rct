"""Collect results/*.json into a tidy table, refusing to proceed on a broken matrix.

The Friedman test needs a complete design x variant matrix: a single missing cell drops
the whole design. The dangerous failure is not a crash but silent partial data -- if 37 of
40 cells are present, the test might run happily, but the reported N is not the N you think. 
So here we want to determine if the matrix is complete, and refuse to load it if it isn't.

This returns a long DataFrame (one row per design, variant, and seed), plus a report of 
what was excluded and why. Nothing here averages or ranks; that happens in aggregate.py. 
The split matters because every exclusion decision should be visible before any statistic 
is computed.
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

# Fields that must agree across every cell of one design for the pairing to hold. 
# If two cells of the same design differ on any of these, the pairing is broken.
PAIRING_FIELDS = ["d_model", "depth", "lr", "batch_size", "max_len", "n_heads"]
LINFORMER_K_FIELD = "linformer_k"


@dataclass
class CollectionReport:
    """Tracks what was found, what was dropped, and why."""

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


# A result file must carry all of these to uniquely identify its cell.
# Anything missing one is likely a probe or stray JSON sharing the directory.
RESULT_SIGNATURE = ["run_id", "phase", "task", "design_id", "variant", "seed"]


def _is_result_file(row: dict) -> bool:
    """Checks if this payload is an actual training result."""
    return all(row.get(field) is not None for field in RESULT_SIGNATURE)


def _load_one(path: Path) -> dict:
    """Reads one result file and flattens the necessary fields into a flat row."""
    payload = json.loads(path.read_text())
    config = payload.get("config", {})
    
    row = {
        "run_id": payload.get("run_id"),
        "phase": payload.get("phase"),
        # Fall back to config then historical default for older ListOps runs
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
        "target_epochs": config.get("epochs"),
        "_source": path.name,
    }
    
    for f in PAIRING_FIELDS:
        row[f] = config.get(f)
    row[LINFORMER_K_FIELD] = config.get(LINFORMER_K_FIELD)
    
    return row


def load_results(results_dir, phase="full") -> tuple[pd.DataFrame, CollectionReport]:
    """Loads every result file for a given phase into a long DataFrame.

    Args:
        results_dir: Directory containing the JSON result files.
        phase: Filters rows so throwaway smoke runs don't contaminate the report.

    Returns:
        (df, report). The report records completeness and integrity findings. 
        It does NOT raise errors here -- `require_complete` handles that.
    """
    results_dir = Path(results_dir)
    report = CollectionReport()
    rows = []
    
    # We recognize result files by their payload rather than a rigid filename glob.
    # This prevents new tasks (like Pathfinder) from being silently ignored.
    for path in sorted(results_dir.glob("*.json")):
        try:
            row = _load_one(path)
        except json.JSONDecodeError as err:
            report.warnings.append(f"could not parse {path.name}: {err}")
            continue
            
        if not _is_result_file(row):
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
    _check_pairing(df, report)
    _check_params(df, report)
    _check_backend(df, report)
    
    return df, report


def _blocks(df):
    """Yields (task, design_id) block keys present in the table in a stable order."""
    seen = df[["task", "design_id"]].drop_duplicates().sort_values(["task", "design_id"])
    for _, r in seen.iterrows():
        yield r["task"], int(r["design_id"])


def _check_completeness(df, report):
    """Records which (task, design) blocks have all 5 variants x 3 seeds."""
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


def _check_partial_training(df, report):
    """Flags cells whose timing reflects fewer epochs than intended.

    Requeued runs report timing over only THIS session's epochs. This doesn't corrupt 
    accuracy, but it skews efficiency numbers, so we flag it instead of silently averaging.
    """
    for _, r in df.iterrows():
        trained, target = r["epochs_trained_this_session"], r["target_epochs"]
        if trained is not None and target is not None and trained < target:
            report.warnings.append(
                f"{r['run_id']}: trained {trained} of {target} epochs this session; "
                "efficiency numbers may cover a partial run"
            )


def _check_pairing(df, report):
    """Ensures every cell in a block agrees on the pairing fields to remain valid."""
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
    """Verifies non-Linformer arms share a param count and Linformer has its expected overhead."""
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
            k_values = block[LINFORMER_K_FIELD].dropna().unique()
            max_len_values = block["max_len"].dropna().unique()
            
            if len(k_values) == 1 and len(max_len_values) == 1:
                expected = int(k_values[0]) * int(max_len_values[0])
                if gap != expected:
                    report.integrity_failures.append(
                        f"{tag}: linformer overhead is {gap:,}, "
                        f"expected {expected:,} (k={int(k_values[0])} x max_len={int(max_len_values[0])})"
                    )
            elif gap <= 0:
                report.integrity_failures.append(
                    f"{tag}: linformer overhead is {gap:,}; expected a positive "
                    "k * max_len and could not verify the exact value"
                )


def _check_backend(df, report):
    """Ensures every Flash cell took the verified FA-2 path."""
    flash = df[df["variant"] == "flash"]
    for _, r in flash.iterrows():
        backend = r["backend"] or ""
        if "VERIFIED" not in backend:
            report.integrity_failures.append(
                f"{r['run_id']}: flash backend not verified (backend={backend!r}); "
                "it may have run a fallback kernel"
            )


def require_complete(df, report, drop_incomplete=True):
    """Returns only analysable designs, raising errors if integrity checks fail.

    Args:
        df: The long DataFrame from load_results.
        report: The corresponding CollectionReport.
        drop_incomplete: Silently drops incomplete designs if True. Raises error if False.
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