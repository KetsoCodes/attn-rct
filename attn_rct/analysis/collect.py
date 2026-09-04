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
            f"complete designs     : {len(self.complete_designs)} "
            f"{sorted(self.complete_designs)}",
        ]
        if self.incomplete_designs:
            lines.append(f"incomplete designs   : {len(self.incomplete_designs)}")
            for design_id, missing in sorted(self.incomplete_designs.items()):
                lines.append(f"    d{design_id:03d} missing: {missing}")
        if self.integrity_failures:
            lines.append("integrity failures   :")
            lines.extend(f"    {f}" for f in self.integrity_failures)
        if self.warnings:
            lines.append("warnings             :")
            lines.extend(f"    {w}" for w in self.warnings)
        return "\n".join(lines)


def _load_one(path: Path) -> dict:
    """Read one result file and flatten the fields the analysis needs into a flat row."""
    payload = json.loads(path.read_text())
    config = payload.get("config", {})
    row = {
        "run_id": payload.get("run_id"),
        "phase": payload.get("phase"),
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
    for path in sorted(results_dir.glob("d*.json")):
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


def _check_completeness(df, report):
    """Record which designs have all 5 variants x 3 seeds, and what the rest are missing."""
    expected = {(v, s) for v in VARIANTS for s in SEEDS}
    for design_id in sorted(df["design_id"].unique()):
        present = set(map(tuple,
                          df[df["design_id"] == design_id][["variant", "seed"]].values))
        missing = expected - present
        if missing:
            report.incomplete_designs[int(design_id)] = sorted(
                f"{v}_s{s}" for v, s in missing
            )
        else:
            report.complete_designs.append(int(design_id))


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
    """Every cell of one design must agree on the pairing fields, or the design is invalid."""
    for design_id in sorted(df["design_id"].unique()):
        block = df[df["design_id"] == design_id]
        for f in PAIRING_FIELDS:
            distinct = block[f].dropna().unique()
            if len(distinct) > 1:
                report.integrity_failures.append(
                    f"d{int(design_id):03d}: {f} varies within design "
                    f"({sorted(distinct)}); pairing is broken"
                )


def _check_params(df, report):
    """Within a design, non-linformer arms share a param count; linformer is +512,000."""
    for design_id in sorted(df["design_id"].unique()):
        block = df[df["design_id"] == design_id]
        others = block[block["variant"] != "linformer"]["n_params"].dropna().unique()
        if len(others) > 1:
            report.integrity_failures.append(
                f"d{int(design_id):03d}: non-linformer arms disagree on n_params "
                f"({sorted(others)}); a config drifted"
            )
        linf = block[block["variant"] == "linformer"]["n_params"].dropna().unique()
        if len(linf) > 1:
            report.integrity_failures.append(
                f"d{int(design_id):03d}: linformer arms disagree on n_params "
                f"({sorted(int(x) for x in linf)}); a config drifted"
            )
        if len(others) == 1 and len(linf) == 1:
            gap = int(linf[0]) - int(others[0])
            if gap != LINFORMER_OVERHEAD:
                report.integrity_failures.append(
                    f"d{int(design_id):03d}: linformer overhead is {gap:,}, "
                    f"expected {LINFORMER_OVERHEAD:,}"
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
            "incomplete designs present and drop_incomplete=False:\n  "
            + "\n  ".join(f"d{d:03d}: {m}"
                          for d, m in sorted(report.incomplete_designs.items()))
        )
    keep = df[df["design_id"].isin(report.complete_designs)].copy()
    if keep.empty:
        raise ValueError(
            "no complete design x variant matrix available; "
            f"complete designs: {report.complete_designs}"
        )
    return keep
