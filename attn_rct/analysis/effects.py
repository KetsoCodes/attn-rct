"""Computes effect sizes to quantify the magnitude of differences between variants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MAGNITUDE_BANDS = [(0.2, "negligible"), (0.5, "small"), (0.8, "medium")]


@dataclass
class EffectResult:
    """Results containing effect sizes for variants compared against a baseline."""

    metric: str
    baseline: str
    n_blocks: int
    direction: str
    table: pd.DataFrame
    n_bootstrap: int

    def summary(self) -> str:
        """Generates a formatted text summary of the effect size results."""
        head = (f"metric   : {self.metric}  ({self.direction} is better)\n"
                f"baseline : {self.baseline}   N={self.n_blocks} blocks   "
                f"{self.n_bootstrap:,} bootstrap resamples")
        columns = [c for c in
                   ["variant", "mean_difference", "cohens_dz", "ci_low", "ci_high",
                    "magnitude", "geometric_ratio", "favours"]
                   if c in self.table.columns]
        body = self.table[columns].to_string(
            index=False, float_format=lambda v: f"{v:.4f}")
        return f"{head}\n{body}"


def describe_magnitude(dz: float) -> str:
    """Categorizes the magnitude of an effect size using conventional thresholds."""
    magnitude = abs(dz)
    for threshold, label in _MAGNITUDE_BANDS:
        if magnitude < threshold:
            return label
    return "large"


def cohens_dz(differences: np.ndarray) -> float:
    """Calculates the paired Cohen's d_z for block-level differences."""
    differences = np.asarray(differences, dtype=float)
    sd = differences.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("inf") if differences.mean() != 0 else 0.0
    return float(differences.mean() / sd)


def bootstrap_ci(differences: np.ndarray, statistic, n_bootstrap=10_000,
                 alpha=0.05, seed=0) -> tuple[float, float]:
    """Calculates a percentile bootstrap confidence interval by resampling blocks."""
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    if n < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        resample = differences[rng.integers(0, n, size=n)]
        estimates[i] = statistic(resample)

    finite = estimates[np.isfinite(estimates)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(finite, 100 * alpha / 2)),
            float(np.percentile(finite, 100 * (1 - alpha / 2))))


def geometric_mean_ratio(baseline_values: np.ndarray,
                         variant_values: np.ndarray) -> float:
    """Calculates the geometric mean of the ratio between baseline and variant values."""
    baseline_values = np.asarray(baseline_values, dtype=float)
    variant_values = np.asarray(variant_values, dtype=float)
    if np.any(baseline_values <= 0) or np.any(variant_values <= 0):
        return float("nan")
    return float(np.exp(np.mean(np.log(baseline_values / variant_values))))


def compute_effects(value_matrix: pd.DataFrame, baseline: str, metric: str,
                    direction: str, n_bootstrap=10_000, alpha=0.05,
                    seed=0, ratio_scale=None) -> EffectResult:
    """Computes effect sizes and confidence intervals for every variant against the baseline."""
    if baseline not in value_matrix.columns:
        raise ValueError(
            f"baseline {baseline!r} not among variants {sorted(value_matrix.columns)}"
        )
    if value_matrix.isna().any().any():
        raise ValueError("value matrix contains NaN; the matrix is incomplete")
    if len(value_matrix) < 2:
        raise ValueError(f"need at least 2 blocks, got {len(value_matrix)}")
    if direction not in ("higher", "lower"):
        raise ValueError(f"direction must be 'higher' or 'lower', got {direction!r}")

    ratio_scale = (direction == "lower") if ratio_scale is None else ratio_scale
    base = value_matrix[baseline].to_numpy(dtype=float)

    rows = []
    for variant in value_matrix.columns:
        if variant == baseline:
            continue
        values = value_matrix[variant].to_numpy(dtype=float)
        differences = values - base

        dz = cohens_dz(differences)
        low, high = bootstrap_ci(differences, cohens_dz,
                                 n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)

        mean_difference = float(differences.mean())
        if mean_difference == 0:
            favours = "neither"
        elif (mean_difference > 0) == (direction == "higher"):
            favours = variant
        else:
            favours = baseline

        row = {
            "variant": variant,
            "mean_difference": mean_difference,
            "cohens_dz": dz,
            "ci_low": low,
            "ci_high": high,
            "magnitude": describe_magnitude(dz) if np.isfinite(dz) else "perfectly consistent",
            "favours": favours,
            "ci_excludes_zero": bool(np.isfinite(low) and np.isfinite(high)
                                     and (low > 0 or high < 0)),
        }
        if ratio_scale:
            row["geometric_ratio"] = geometric_mean_ratio(base, values)
        rows.append(row)

    table = pd.DataFrame(rows)
    table = table.reindex(
        table["cohens_dz"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    return EffectResult(metric=metric, baseline=baseline, n_blocks=len(value_matrix),
                        direction=direction, table=table, n_bootstrap=n_bootstrap)