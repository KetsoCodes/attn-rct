"""Collects and validates experimental result files into a structured dataset.

Ensures data integrity by verifying that only completely executed experimental blocks 
(all variants and seeds for a given task and design) are passed forward. This stage 
strictly flags or drops incomplete or corrupted matrices before statistical analysis begins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

VARIANTS = ["vanilla", "flash", "linformer", "linear", "sparse"]
SEEDS = [0, 1, 2]
N_DESIGNS = 8
LINFORMER_OVERHEAD = 512_000

PAIRING_FIELDS = ["d_model", "depth", "lr", "batch_size", "max_len", "n_heads"]


@dataclass
class CollectionReport:
    """Tracks the status of loaded results, including dropped blocks and integrity warnings."""

    n_files: int = 0
    n_rows: int = 0
    complete_designs: list = field(default_factory=list)
    incomplete_designs: dict = field(default_factory=dict)
    integrity_failures: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        """Generates a formatted summary of the collection process."""
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


def _load_one(path: Path) -> dict:
    """Parses a single JSON result file into a flat dictionary."""
    payload = json.loads(path.read_text())
    config = payload.get("config", {})
    row = {
        "run_id": payload.get("run_id"),
        "phase": payload.get("phase"),
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
    return row


def load_results(results_dir, phase="full") -> tuple[pd.DataFrame, CollectionReport]:
    """Loads all valid result files for a given phase into a DataFrame.

    Args:
        results_dir: Directory containing the JSON result files.
        phase: Filters rows by phase to exclude throwaway/smoke runs.

    Returns:
        A tuple containing the results DataFrame and a CollectionReport detailing 
        any integrity or completeness findings.
    """
    results_dir = Path(results_dir)
    report = CollectionReport()

    rows = []
    patterns = ["listops_*.json", "cifar_*.json", "d[0-9]*.json"]
    candidates = sorted(
        {path for pattern in patterns for path in results_dir.glob(pattern)}
    )
    
    for path in candidates:
        try:
            row = _load_one(path)
        except (json.JSONDecodeError, KeyError) as err:
            report.warnings.append(f"could not parse {path.name}: {err}")
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
    """Yields unique (task, design_id) combinations present in the dataset."""
    seen = df[["task", "design_id"]].drop_duplicates()
    seen = seen.sort_values(["task", "design_id"])
    for _, r in seen.iterrows():
        yield r["task"], int(r["design_id"])


def _check_completeness(df, report):
    """Identifies blocks containing the full expected matrix of variants and seeds."""
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
    """Flags runs that completed fewer epochs than targeted, as timing metrics may be skewed."""
    for _, r in df.iterrows():
        trained, target = r["epochs_trained_this_session"], r["target_epochs"]
        if trained is not None and target is not None and trained < target:
            report.warnings.append(
                f"{r['run_id']}: trained {trained} of {target} epochs this session; "
                "efficiency numbers may cover a partial run"
            )


def _check_pairing(df, report):
    """Ensures all runs within a block share identical architectural configurations."""
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
    """Verifies parameter count consistency within blocks, accounting for Linformer's overhead."""
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
            if gap != LINFORMER_OVERHEAD:
                report.integrity_failures.append(
                    f"{tag}: linformer overhead is {gap:,}, "
                    f"expected {LINFORMER_OVERHEAD:,}"
                )


def _check_backend(df, report):
    """Ensures all FlashAttention runs utilized the verified hardware-accelerated backend."""
    flash = df[df["variant"] == "flash"]
    for _, r in flash.iterrows():
        backend = r["backend"] or ""
        if "VERIFIED" not in backend:
            report.integrity_failures.append(
                f"{r['run_id']}: flash backend not verified (backend={backend!r}); "
                "it may have run a fallback kernel"
            )


def require_complete(df, report, drop_incomplete=True):
    """Filters the DataFrame to return only fully complete and intact experimental blocks.

    Raises:
        ValueError: On any integrity failure, or if no complete designs remain.
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