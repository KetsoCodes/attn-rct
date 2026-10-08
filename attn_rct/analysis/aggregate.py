"""Collapse seeds into cells and rank variants within each block.

This is the stage where the paired design either holds or is quietly lost.

A seed is noise control, not an observation. Three seeds of one (design, variant) are
three re-rolls of the same experiment, so they are averaged into a single cell. If they
entered the Friedman test as separate rows, N would appear to be 24 rather than 8, the
test would reject far more readily than it should, and the rejection would be
pseudoreplication -- a claim about the design space supported by nothing more than
repeated draws from the same design.

Ranking happens WITHIN a block rather than across the whole table. A deep, wide design is
slower for every arm, so its absolute seconds are not comparable to a small design's.
Ranking within the block removes that scale and keeps only the ordering, which is the
quantity the study generalises over.

A block is (task, design). With a single task that is just the design; when the task
suite grows, the same code yields task x design blocks with no change, and the per-task
breakdown -- does the ranking CHANGE by task? -- becomes available for free.
"""

from __future__ import annotations

import pandas as pd

# Direction is a property of the metric, not a guess made at ranking time. Getting it
# backwards inverts every conclusion while still producing a plausible-looking diagram.
# "higher" means a larger value is better; "lower" means a smaller value is better.
METRIC_DIRECTION = {
    "best_val_accuracy": "higher",
    "final_val_accuracy": "higher",
    "mean_train_seconds_per_epoch": "lower",
    "peak_memory_mb": "lower",
    "n_params": "lower",
}

BLOCK_KEYS = ["task", "design_id"]

# Chance accuracy per task, from each task's class count. Kept here rather than imported
# from attn_rct.data so the analysis runs without torch, on a laptop.
TASK_CHANCE = {
    "listops": 1 / 10,
    "cifar": 1 / 10,
    "pathfinder": 1 / 2,
    "text": 1 / 2,
    "retrieval": 1 / 2,
}

# How far above chance one arm must reach for a block to count as informative. Fixed at
# 0.02 in analysis_plan_amendment_01 before any ranking was examined; it is four times the
# largest excursion ever observed in a degenerate Pathfinder block (0.005).
DEGENERATE_MARGIN = 0.02

# Only accuracy can be degenerate in this sense. Seconds and megabytes are measured
# whether or not the model learned, which is exactly why Pathfinder still contributes
# efficiency blocks after failing to produce accuracy blocks.
ACCURACY_METRICS = ["best_val_accuracy", "final_val_accuracy"]


def _ensure_task(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a task column so blocks are well defined even for a single-task run."""
    if "task" not in df.columns:
        df = df.copy()
        df["task"] = "listops"
    return df


def aggregate_seeds(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Average seeds within each (block, variant) cell.

    Args:
        df: long table from collect.load_results, one row per (design, variant, seed).
        metrics: metric columns to aggregate. Defaults to every known metric present.

    Returns:
        One row per (task, design_id, variant), with <metric> holding the seed mean and
        <metric>_sd the seed standard deviation, plus n_seeds.

    Raises:
        ValueError: if a requested metric is absent, or a cell has no usable values.
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


def rank_within_blocks(cells: pd.DataFrame, metric: str,
                       direction: str | None = None) -> pd.DataFrame:
    """Rank variants within each block, 1 = best, ties share the average rank.

    Args:
        cells: output of aggregate_seeds.
        metric: the column to rank on.
        direction: "higher" or "lower". Defaults to METRIC_DIRECTION[metric].

    Returns:
        cells with a `rank` column added.

    Raises:
        ValueError: on an unknown metric with no declared direction, or if the blocks do
            not all contain the same set of variants -- an incomplete block cannot be
            ranked, and silently ranking 4 arms where others have 5 would corrupt the
            mean ranks without any visible error.
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
            "blocks contain different numbers of variants; the matrix is incomplete "
            f"({sizes.to_dict()})"
        )

    out = cells.copy()
    # ascending=True gives rank 1 to the smallest value, which is what "lower is better"
    # wants; "higher is better" is the same thing on the negated column.
    ascending = direction == "lower"
    out["rank"] = (
        out.groupby(BLOCK_KEYS)[metric]
        .rank(method="average", ascending=ascending)
    )
    return out


def degenerate_blocks(cells: pd.DataFrame, metric: str, margin=DEGENERATE_MARGIN):
    """Find blocks in which no arm beat chance, so the block carries no ranking.

    Why these have to come out of the omnibus rather than be left in as "more data": a
    block where all five arms tie contributes the identical rank vector (3, 3, 3, 3, 3).
    It adds nothing to the Friedman statistic's numerator, but it still increments the
    number of blocks the statistic is divided through by, so including it *lowers* the
    power of the test on the blocks that do carry an effect. Leaving them in would look
    thorough and silently weaken the headline comparison.

    This applies to accuracy only. An efficiency metric is measured whether or not the
    model learned anything, which is why a task can fail to produce accuracy blocks and
    still contribute efficiency blocks from the same runs.

    Args:
        cells: seed-averaged cells from aggregate_seeds.
        metric: the metric being ranked; non-accuracy metrics always return nothing.
        margin: how far above chance the best arm must reach.

    Returns:
        A list of (task, design_id) block keys that are degenerate, in sorted order.
    """
    if metric not in ACCURACY_METRICS:
        return []

    out = []
    for block, group in _ensure_task(cells).groupby(BLOCK_KEYS, sort=True):
        task = block[0]
        chance = TASK_CHANCE.get(task)
        if chance is None:
            # An unregistered task is not assumed degenerate -- silently dropping blocks
            # for an unknown task would be the same class of bug as the collector's old
            # per-task glob. Better to keep it and have the omnibus report the tie.
            continue
        if group[metric].max() <= chance + margin:
            out.append(block)
    return out


def drop_degenerate_blocks(cells: pd.DataFrame, metric: str,
                           margin=DEGENERATE_MARGIN):
    """Remove degenerate blocks, returning the survivors and what was removed.

    Returns:
        (kept, removed) where removed is the list of (task, design_id) block keys. The
        caller is expected to report `removed` -- an exclusion that is not stated is
        indistinguishable from data that never existed.
    """
    cells = _ensure_task(cells)
    removed = degenerate_blocks(cells, metric, margin)
    if not removed:
        return cells, []
    keys = set(removed)
    mask = [
        (task, design) not in keys
        for task, design in zip(cells["task"], cells["design_id"])
    ]
    return cells.loc[mask].copy(), removed


def rank_matrix(ranked: pd.DataFrame) -> pd.DataFrame:
    """Pivot to the blocks x variants rank matrix the omnibus test consumes.

    Returns:
        DataFrame indexed by block, one column per variant, values are ranks.
    """
    return ranked.pivot_table(index=BLOCK_KEYS, columns="variant", values="rank")


def value_matrix(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot to the blocks x variants matrix of raw cell means, for effect sizes."""
    return cells.pivot_table(index=BLOCK_KEYS, columns="variant", values=metric)


def mean_ranks(ranked: pd.DataFrame) -> pd.Series:
    """Mean rank per variant across blocks, ascending so the best arm is first."""
    return ranked.groupby("variant")["rank"].mean().sort_values()


def seed_noise_report(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compare within-cell seed noise against between-variant spread in the same block.

    If seed noise is the same order as the differences between arms, three seeds is too
    few for this metric to be ranked reliably, and the pilot should say so rather than
    quietly reporting a ranking built on noise.

    Returns:
        One row per block with the mean seed SD, the spread across variants, and their
        ratio. A ratio near or above 1 is a warning.
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
