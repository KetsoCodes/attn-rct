"""Unit tests for the aggregation and ranking stage of the analysis pipeline.

Ensures that random seeds are properly collapsed to prevent pseudoreplication, 
and verifies that intra-block ranking correctly applies metric directionality, 
handles tied values, and supports multi-task blocking.
"""

import pandas as pd
import pytest

from attn_rct.analysis import aggregate, collect, synth


@pytest.fixture
def cells(tmp_path):
    """Provides a complete synthetic grid, collected and collapsed to cells."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    df, report = collect.load_results(tmp_path, phase="full")
    keep = collect.require_complete(df, report)
    return aggregate.aggregate_seeds(keep)


def test_seeds_collapse_to_one_row_per_cell(cells):
    """Verifies that 120 runs are properly collapsed into 40 cells (8 designs x 5 variants)."""
    assert len(cells) == 40
    assert (cells["n_seeds"] == 3).all()
    assert cells.groupby(["task", "design_id"])["variant"].nunique().eq(5).all()


def test_seed_mean_is_actually_the_mean(tmp_path):
    """Ensures the aggregated cell value is the mathematical mean of its constituent seeds."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    df, report = collect.load_results(tmp_path, phase="full")
    keep = collect.require_complete(df, report)
    cells = aggregate.aggregate_seeds(keep)

    raw = keep[(keep["design_id"] == 0) & (keep["variant"] == "flash")]
    cell = cells[(cells["design_id"] == 0) & (cells["variant"] == "flash")]
    assert cell["best_val_accuracy"].iloc[0] == pytest.approx(
        raw["best_val_accuracy"].mean()
    )


def test_lower_is_better_ranks_smallest_first(cells):
    """Checks that for 'lower-is-better' metrics (e.g., memory), the smallest value ranks 1."""
    ranked = aggregate.rank_within_blocks(cells, "peak_memory_mb")
    for _, block in ranked.groupby(["task", "design_id"]):
        cheapest = block.loc[block["peak_memory_mb"].idxmin(), "variant"]
        assert block.loc[block["variant"] == cheapest, "rank"].iloc[0] == 1.0


def test_higher_is_better_ranks_largest_first(cells):
    """Checks that for 'higher-is-better' metrics (e.g., accuracy), the largest value ranks 1."""
    ranked = aggregate.rank_within_blocks(cells, "best_val_accuracy")
    for _, block in ranked.groupby(["task", "design_id"]):
        best = block.loc[block["best_val_accuracy"].idxmax(), "variant"]
        assert block.loc[block["variant"] == best, "rank"].iloc[0] == 1.0


def test_direction_flip_reverses_the_ranking(cells):
    """Ensures that reversing the evaluation direction perfectly mirrors the resulting ranks."""
    k = cells["variant"].nunique()
    lower = aggregate.rank_within_blocks(cells, "peak_memory_mb", direction="lower")
    higher = aggregate.rank_within_blocks(cells, "peak_memory_mb", direction="higher")
    merged = lower.merge(
        higher, on=["task", "design_id", "variant"], suffixes=("_low", "_high")
    )
    assert (merged["rank_low"] + merged["rank_high"] == k + 1).all()


def test_ties_share_the_average_rank():
    """Verifies that tied metrics correctly share the average of the ranks they span."""
    cells = pd.DataFrame({
        "task": ["listops"] * 4,
        "design_id": [0] * 4,
        "variant": ["a", "b", "c", "d"],
        "peak_memory_mb": [10.0, 20.0, 20.0, 30.0],
    })
    ranked = aggregate.rank_within_blocks(cells, "peak_memory_mb")
    by_variant = dict(zip(ranked["variant"], ranked["rank"]))
    assert by_variant["a"] == 1.0
    assert by_variant["b"] == by_variant["c"] == 2.5
    assert by_variant["d"] == 4.0


def test_ranking_is_within_block_not_global(cells):
    """Ensures rankings are strictly contained within their block (1..k)."""
    ranked = aggregate.rank_within_blocks(cells, "mean_train_seconds_per_epoch")
    k = cells["variant"].nunique()
    for _, block in ranked.groupby(["task", "design_id"]):
        assert sorted(block["rank"]) == [float(i) for i in range(1, k + 1)]


def test_incomplete_block_is_refused(cells):
    """Verifies that ranking is rejected if a block is missing one or more variants."""
    broken = cells[~((cells["design_id"] == 2) & (cells["variant"] == "sparse"))]
    with pytest.raises(ValueError, match="incomplete"):
        aggregate.rank_within_blocks(broken, "peak_memory_mb")


def test_unknown_metric_needs_explicit_direction(cells):
    """Ensures ranking fails if the metric does not have an explicitly configured direction."""
    cells = cells.copy()
    cells["mystery_metric"] = 1.0
    with pytest.raises(ValueError, match="no direction declared"):
        aggregate.rank_within_blocks(cells, "mystery_metric")


def test_rank_matrix_shape(cells):
    """Validates the structure of the pivoted matrix required for the omnibus test."""
    ranked = aggregate.rank_within_blocks(cells, "best_val_accuracy")
    matrix = aggregate.rank_matrix(ranked)
    assert matrix.shape == (8, 5)
    assert not matrix.isna().any().any()


def test_missing_metric_column_is_an_error(cells):
    """Ensures the aggregation step fails immediately if requested metrics are missing."""
    with pytest.raises(ValueError, match="not present"):
        aggregate.aggregate_seeds(cells, metrics=["nonexistent_metric"])


def test_seed_noise_report_flags_high_noise():
    """Checks that the noise report correctly calculates the ratio of seed SD to variant spread."""
    noisy = pd.DataFrame({
        "task": ["listops"] * 3,
        "design_id": [0] * 3,
        "variant": ["a", "b", "c"],
        "best_val_accuracy": [0.30, 0.31, 0.32],
        "best_val_accuracy_sd": [0.02, 0.02, 0.02],
    })
    report = aggregate.seed_noise_report(noisy, "best_val_accuracy")
    assert report["noise_ratio"].iloc[0] == pytest.approx(1.0, rel=1e-6)


def test_blocks_are_task_aware(cells):
    """Verifies that multi-task data expands the matrix block rows natively."""
    two_tasks = pd.concat([
        cells.assign(task="listops"),
        cells.assign(task="pathfinder"),
    ], ignore_index=True)
    ranked = aggregate.rank_within_blocks(two_tasks, "peak_memory_mb")
    matrix = aggregate.rank_matrix(ranked)
    assert matrix.shape == (16, 5)