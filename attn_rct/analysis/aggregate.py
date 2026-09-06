"""Aggregates random seeds into cell averages and ranks attention variants within architectural blocks.

This stage prepares the raw data for statistical testing by averaging out initialization noise
and ranking the variants relative to their specific design constraints.
"""

from __future__ import annotations

import pandas as pd

# Defines whether a higher or lower value indicates better performance for each metric.
METRIC_DIRECTION = {
    "best_val_accuracy": "higher",
    "final_val_accuracy": "higher",
    "mean_train_seconds_per_epoch": "lower",
    "peak_memory_mb": "lower",
    "n_params": "lower",
}

BLOCK_KEYS = ["task", "design_id"]


def _ensure_task(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a default 'listops' task column if missing to maintain consistent block structures."""
    if "task" not in df.columns:
        df = df.copy()
        df["task"] = "listops"
    return df


def aggregate_seeds(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Averages the results of multiple random seeds for each (block, variant) combination.

    Args:
        df: Long table from collect.load_results.
        metrics: Metric columns to aggregate. Defaults to all known metrics present.

    Returns:
        DataFrame with one row per (task, design_id, variant), containing the mean and 
        standard deviation for each requested metric.
        
    Raises:
        ValueError: If a requested metric is absent or contains unusable values.
    """
    df = _ensure_task(df)
    if metrics is None:
        metrics = [m for m in METRIC_DIRECTION if m in df.columns]
        
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"metrics not present in the table: {missing}")

    grouped = df.groupby(BLOCK_KEYS + ["variant"], as_index=False)
    cells = grouped.agg(
        n_seeds=("seed", "count"),
        **{m: (m, "mean") for m in metrics},
        **{f"{m}_sd": (m, "std") for m in metrics},
    )

    for m in metrics:
        if cells[m].isna().any():
            bad = cells[cells[m].isna()][BLOCK_KEYS + ["variant"]].to_dict("records")
            raise ValueError(f"metric {m!r} is missing for cells: {bad}")
            
    return cells


def rank_within_blocks(cells: pd.DataFrame, metric: str, direction: str | None = None) -> pd.DataFrame:
    """Ranks attention variants within each block (1 = best). Ties share the average rank.

    Args:
        cells: Output DataFrame from aggregate_seeds.
        metric: The column to rank on.
        direction: "higher" or "lower". Defaults to METRIC_DIRECTION[metric].

    Returns:
        DataFrame with a new `rank` column.

    Raises:
        ValueError: If the metric direction is unknown or if blocks have mismatched variant counts.
    """
    direction = direction or METRIC_DIRECTION.get(metric)
    if direction not in ("higher", "lower"):
        raise ValueError(
            f"no direction declared for metric {metric!r}; pass direction= explicitly "
            f"(known: {sorted(METRIC_DIRECTION)})"
        )

    sizes = cells.groupby(BLOCK_KEYS)["variant"].nunique()
    if sizes.nunique() != 1:
        raise ValueError(
            f"blocks contain different numbers of variants; the matrix is incomplete ({sizes.to_dict()})"
        )

    out = cells.copy()
    ascending = direction == "lower"
    out["rank"] = out.groupby(BLOCK_KEYS)[metric].rank(method="average", ascending=ascending)
    
    return out


def rank_matrix(ranked: pd.DataFrame) -> pd.DataFrame:
    """Pivots the data into a (blocks x variants) matrix of ranks for the omnibus test."""
    return ranked.pivot_table(index=BLOCK_KEYS, columns="variant", values="rank")


def value_matrix(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivots the data into a (blocks x variants) matrix of raw metric means."""
    return cells.pivot_table(index=BLOCK_KEYS, columns="variant", values=metric)


def mean_ranks(ranked: pd.DataFrame) -> pd.Series:
    """Calculates the average rank per variant across all blocks, sorted best to worst."""
    return ranked.groupby("variant")["rank"].mean().sort_values()


def seed_noise_report(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Evaluates if seed variance is too high compared to performance differences between variants.

    Returns:
        DataFrame containing a noise_ratio. A ratio near or above 1.0 indicates the ranking 
        is driven by initialization noise rather than architectural differences.
    """
    rows = []
    for block, group in cells.groupby(BLOCK_KEYS):
        seed_sd = group[f"{metric}_sd"].mean()
        spread = group[metric].max() - group[metric].min()
        rows.append({
            **dict(zip(BLOCK_KEYS, block if isinstance(block, tuple) else (block,))),
            "mean_seed_sd": seed_sd,
            "variant_spread": spread,
            "noise_ratio": seed_sd / spread if spread else float("inf"),
        })
        
    return pd.DataFrame(rows)