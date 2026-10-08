"""Tests for the aggregation and ranking stage.

Two properties matter more than the rest, because getting either wrong produces output
that looks entirely reasonable:

    Seeds must collapse. If they leak through as separate rows, N inflates threefold and
    every downstream p-value is wrong in the direction of rejecting.

    Direction must be honoured per metric. Ranking seconds as though higher were better
    inverts the conclusion while still drawing a plausible diagram.

So both are tested directly rather than inferred from a happy-path run.
"""

import pandas as pd
import pytest

from attn_rct.analysis import aggregate, collect, synth


@pytest.fixture
def cells(tmp_path):
    """A complete synthetic grid, collected and collapsed to cells."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    df, report = collect.load_results(tmp_path, phase="full")
    keep = collect.require_complete(df, report)
    return aggregate.aggregate_seeds(keep)


def test_seeds_collapse_to_one_row_per_cell(cells):
    """120 runs become 40 cells: 8 designs x 5 variants, seeds averaged."""
    assert len(cells) == 40
    assert (cells["n_seeds"] == 3).all()
    assert cells.groupby(["task", "design_id"])["variant"].nunique().eq(5).all()


def test_seed_mean_is_actually_the_mean(tmp_path):
    """The cell value is the arithmetic mean of its seeds, not the first or the best."""
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
    """On a lower-is-better metric the cheapest arm gets rank 1."""
    ranked = aggregate.rank_within_blocks(cells, "peak_memory_mb")
    for _, block in ranked.groupby(["task", "design_id"]):
        cheapest = block.loc[block["peak_memory_mb"].idxmin(), "variant"]
        assert block.loc[block["variant"] == cheapest, "rank"].iloc[0] == 1.0


def test_higher_is_better_ranks_largest_first(cells):
    """On a higher-is-better metric the most accurate arm gets rank 1."""
    ranked = aggregate.rank_within_blocks(cells, "best_val_accuracy")
    for _, block in ranked.groupby(["task", "design_id"]):
        best = block.loc[block["best_val_accuracy"].idxmax(), "variant"]
        assert block.loc[block["variant"] == best, "rank"].iloc[0] == 1.0


def test_direction_flip_reverses_the_ranking(cells):
    """The same data ranked the other way must produce exactly mirrored ranks.

    With k variants, rank r under one direction must be k+1-r under the other. This is
    the check that would have caught a direction bug in either the metric table or the
    ranking call.
    """
    k = cells["variant"].nunique()
    lower = aggregate.rank_within_blocks(cells, "peak_memory_mb", direction="lower")
    higher = aggregate.rank_within_blocks(cells, "peak_memory_mb", direction="higher")
    merged = lower.merge(
        higher, on=["task", "design_id", "variant"], suffixes=("_low", "_high")
    )
    assert (merged["rank_low"] + merged["rank_high"] == k + 1).all()


def test_ties_share_the_average_rank():
    """Tied values take the mean of the ranks they span, not an arbitrary order."""
    cells = pd.DataFrame({
        "task": ["listops"] * 4,
        "design_id": [0] * 4,
        "variant": ["a", "b", "c", "d"],
        "peak_memory_mb": [10.0, 20.0, 20.0, 30.0],
    })
    ranked = aggregate.rank_within_blocks(cells, "peak_memory_mb")
    by_variant = dict(zip(ranked["variant"], ranked["rank"]))
    assert by_variant["a"] == 1.0
    assert by_variant["b"] == by_variant["c"] == 2.5   # ranks 2 and 3 averaged
    assert by_variant["d"] == 4.0


def test_ranking_is_within_block_not_global(cells):
    """Every block ranks 1..k independently; a slow design does not push its arms down."""
    ranked = aggregate.rank_within_blocks(cells, "mean_train_seconds_per_epoch")
    k = cells["variant"].nunique()
    for _, block in ranked.groupby(["task", "design_id"]):
        assert sorted(block["rank"]) == [float(i) for i in range(1, k + 1)]


def test_incomplete_block_is_refused(cells):
    """A block missing a variant cannot be ranked; silently ranking 4 of 5 would corrupt
    the mean ranks with no visible error."""
    broken = cells[~((cells["design_id"] == 2) & (cells["variant"] == "sparse"))]
    with pytest.raises(ValueError, match="incomplete"):
        aggregate.rank_within_blocks(broken, "peak_memory_mb")


def test_unknown_metric_needs_explicit_direction(cells):
    """A metric with no declared direction must not be ranked on a guess."""
    cells = cells.copy()
    cells["mystery_metric"] = 1.0
    with pytest.raises(ValueError, match="no direction declared"):
        aggregate.rank_within_blocks(cells, "mystery_metric")


def test_rank_matrix_shape(cells):
    """The pivoted matrix is blocks x variants, which is what the omnibus test consumes."""
    ranked = aggregate.rank_within_blocks(cells, "best_val_accuracy")
    matrix = aggregate.rank_matrix(ranked)
    assert matrix.shape == (8, 5)
    assert not matrix.isna().any().any()


def test_missing_metric_column_is_an_error(cells):
    """Asking for a metric that was never collected fails loudly."""
    with pytest.raises(ValueError, match="not present"):
        aggregate.aggregate_seeds(cells, metrics=["nonexistent_metric"])


def test_seed_noise_report_flags_high_noise():
    """When seed noise rivals the between-variant spread, the ratio approaches 1."""
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
    """Blocks key on (task, design) so a second task adds rows rather than needing a rewrite."""
    two_tasks = pd.concat([
        cells.assign(task="listops"),
        cells.assign(task="pathfinder"),
    ], ignore_index=True)
    ranked = aggregate.rank_within_blocks(two_tasks, "peak_memory_mb")
    matrix = aggregate.rank_matrix(ranked)
    assert matrix.shape == (16, 5)      # 2 tasks x 8 designs


def degenerate_cells():
    """Two tasks: listops blocks carry a real effect, pathfinder blocks sit at chance."""
    rows = []
    for design in range(4):
        for i, variant in enumerate(aggregate_module_variants()):
            rows.append({
                "task": "listops", "design_id": design, "variant": variant,
                "best_val_accuracy": 0.30 + 0.01 * i,
                "mean_train_seconds_per_epoch": 100.0 + 10 * i,
            })
            # Chance on pathfinder is 0.5; these wobble by at most 0.003.
            rows.append({
                "task": "pathfinder", "design_id": design, "variant": variant,
                "best_val_accuracy": 0.500 + 0.001 * (i % 4),
                "mean_train_seconds_per_epoch": 700.0 + 10 * i,
            })
    return pd.DataFrame(rows)


def aggregate_module_variants():
    return ["vanilla", "flash", "linformer", "linear", "sparse"]


def test_degenerate_blocks_are_found_for_accuracy():
    """Blocks where no arm beats chance by the margin are flagged."""
    cells = degenerate_cells()
    found = aggregate.degenerate_blocks(cells, "best_val_accuracy")
    assert found == [("pathfinder", d) for d in range(4)]


def test_learning_blocks_are_not_flagged():
    """A task comfortably above its own chance level is untouched."""
    cells = degenerate_cells()
    found = aggregate.degenerate_blocks(cells, "best_val_accuracy")
    assert not any(task == "listops" for task, _ in found)


def test_chance_is_per_task_not_global():
    """0.30 is excellent on a 10-class task and catastrophic on a binary one.

    A single global chance level would either exonerate a dead binary task or condemn a
    healthy 10-class one, so the threshold has to come from the task's class count.
    """
    rows = [
        {"task": "listops", "design_id": 0, "variant": v,
         "best_val_accuracy": 0.30, "mean_train_seconds_per_epoch": 10.0}
        for v in aggregate_module_variants()
    ] + [
        {"task": "pathfinder", "design_id": 0, "variant": v,
         "best_val_accuracy": 0.30, "mean_train_seconds_per_epoch": 10.0}
        for v in aggregate_module_variants()
    ]
    found = aggregate.degenerate_blocks(pd.DataFrame(rows), "best_val_accuracy")
    # listops: 0.30 >> 0.10 + 0.02, kept. pathfinder: 0.30 <= 0.50 + 0.02, degenerate.
    assert found == [("pathfinder", 0)]


def test_efficiency_metrics_are_never_degenerate():
    """Seconds and megabytes are measured whether or not the model learned.

    This is what lets a task that produced no accuracy signal still contribute its
    efficiency blocks, which is the whole basis of Rule B.
    """
    cells = degenerate_cells()
    assert aggregate.degenerate_blocks(cells, "mean_train_seconds_per_epoch") == []
    assert aggregate.degenerate_blocks(cells, "peak_memory_mb") == []


def test_drop_degenerate_keeps_the_rest_and_reports_what_went():
    cells = degenerate_cells()
    kept, removed = aggregate.drop_degenerate_blocks(cells, "best_val_accuracy")
    assert len(removed) == 4
    assert set(kept["task"]) == {"listops"}
    assert len(kept) == 20                      # 4 designs x 5 arms


def test_drop_is_a_no_op_when_nothing_is_degenerate():
    """The common case must not copy or reorder anything into a different answer."""
    cells = degenerate_cells()
    kept, removed = aggregate.drop_degenerate_blocks(cells, "mean_train_seconds_per_epoch")
    assert removed == []
    assert len(kept) == len(cells)


def test_unregistered_task_is_kept_not_silently_dropped():
    """An unknown task has no chance level, so it must not be assumed degenerate.

    Silently dropping blocks for a task the analysis has not heard of is the same class of
    bug as the collector's old per-task glob: invisible, and it makes the report wrong
    while looking right.
    """
    rows = [
        {"task": "brandnewtask", "design_id": 0, "variant": v,
         "best_val_accuracy": 0.5, "mean_train_seconds_per_epoch": 10.0}
        for v in aggregate_module_variants()
    ]
    assert aggregate.degenerate_blocks(pd.DataFrame(rows), "best_val_accuracy") == []


def test_margin_is_respected_at_the_boundary():
    """An arm exactly at the margin is degenerate; just past it is not."""
    def block(acc):
        return pd.DataFrame([
            {"task": "pathfinder", "design_id": 0, "variant": v,
             "best_val_accuracy": acc if v == "flash" else 0.5,
             "mean_train_seconds_per_epoch": 10.0}
            for v in aggregate_module_variants()
        ])
    margin = aggregate.DEGENERATE_MARGIN
    assert aggregate.degenerate_blocks(block(0.5 + margin), "best_val_accuracy") == [
        ("pathfinder", 0)
    ]
    assert aggregate.degenerate_blocks(block(0.5 + margin + 1e-6),
                                       "best_val_accuracy") == []


def test_one_arm_above_chance_saves_the_whole_block():
    """A block is informative if ANY arm learned -- that is a real ranking to measure."""
    rows = [
        {"task": "pathfinder", "design_id": 0, "variant": v,
         "best_val_accuracy": 0.71 if v == "flash" else 0.5,
         "mean_train_seconds_per_epoch": 10.0}
        for v in aggregate_module_variants()
    ]
    assert aggregate.degenerate_blocks(pd.DataFrame(rows), "best_val_accuracy") == []
