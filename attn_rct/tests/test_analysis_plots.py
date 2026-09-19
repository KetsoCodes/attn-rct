"""Unit tests for the visualization and reporting stage.

Verifies that generated figures render correctly (non-empty files) and that 
summary tables strictly match the underlying statistical results without data drift.
"""

from pathlib import Path

import pandas as pd
import pytest

from attn_rct.analysis import (
    aggregate, collect, effects, omnibus, plots, posthoc, synth
)


@pytest.fixture
def analysed(tmp_path):
    """Generates a complete set of statistical results (omnibus, post-hoc, effects) for testing."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    df, report = collect.load_results(tmp_path, phase="full")
    cells = aggregate.aggregate_seeds(collect.require_complete(df, report))
    
    metric = "peak_memory_mb"
    ranked = aggregate.rank_within_blocks(cells, metric)
    rm = aggregate.rank_matrix(ranked)
    mr = aggregate.mean_ranks(ranked)
    
    om = omnibus.run_omnibus(rm, metric, n_permutations=1000)
    ph = posthoc.compare_to_control(mr, "flash", len(rm), metric, method="finner")
    ef = effects.compute_effects(
        aggregate.value_matrix(cells, metric), "flash", metric, "lower", n_bootstrap=500
    )
    
    return {"cells": cells, "mr": mr, "om": om, "ph": ph, "ef": ef, "metric": metric}


def test_cd_diagram_writes_a_nonempty_file(analysed, tmp_path):
    """Ensures CD diagrams render to valid, non-empty SVG files."""
    out = tmp_path / "cd.svg"
    path = plots.critical_difference_diagram(
        analysed["mr"], analysed["ph"], "test", out
    )
    assert path.exists()
    assert path.stat().st_size > 1000


def test_cd_diagram_renders_png_too(analysed, tmp_path):
    """Ensures CD diagrams render to valid, non-empty PNG files."""
    out = tmp_path / "cd.png"
    plots.critical_difference_diagram(
        analysed["mr"], analysed["ph"], "test", out
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_results_table_has_one_row_per_variant(analysed):
    """Verifies the results table includes exactly one row per experimental variant."""
    table = plots.results_table(analysed["om"], analysed["ph"], analysed["ef"])
    assert len(table) == 5
    assert set(table["variant"]) == {"vanilla", "flash", "linformer", "linear", "sparse"}


def test_results_table_marks_the_baseline(analysed):
    """Checks that the baseline variant is explicitly flagged."""
    table = plots.results_table(analysed["om"], analysed["ph"], analysed["ef"])
    baseline_rows = table[table["is_baseline"]]
    assert len(baseline_rows) == 1
    assert baseline_rows.iloc[0]["variant"] == "flash"


def test_results_table_matches_posthoc_pvalues(analysed):
    """Ensures adjusted p-values match the post-hoc computations precisely."""
    table = plots.results_table(analysed["om"], analysed["ph"], analysed["ef"]).set_index("variant")
    ph = analysed["ph"].table.set_index("variant")
    
    for variant in ph.index:
        assert table.loc[variant, "p_adjusted"] == pytest.approx(
            ph.loc[variant, "p_adjusted"], abs=5e-4
        )


def test_results_table_matches_effect_sizes(analysed):
    """Ensures effect sizes (Cohen's dz) match the effects stage precisely."""
    table = plots.results_table(analysed["om"], analysed["ph"], analysed["ef"]).set_index("variant")
    ef = analysed["ef"].table.set_index("variant")
    
    for variant in ef.index:
        assert table.loc[variant, "cohens_dz"] == pytest.approx(
            ef.loc[variant, "cohens_dz"], abs=5e-3
        )


def test_results_table_is_ordered_by_rank(analysed):
    """Verifies the table is sorted by mean rank."""
    table = plots.results_table(analysed["om"], analysed["ph"], analysed["ef"])
    assert list(table["mean_rank"]) == sorted(table["mean_rank"])


def test_cross_task_figure_writes(analysed, tmp_path):
    """Verifies that cross-task comparison figures render to valid files."""
    per_task = {
        "listops": analysed["mr"],
        "cifar": analysed["mr"].sort_index(ascending=False),  # Provide a different ordering
    }
    out = tmp_path / "crosstask.svg"
    path = plots.task_comparison_figure(per_task, analysed["metric"], "test", out)
    assert path.exists()
    assert path.stat().st_size > 1000


def test_cd_diagram_handles_no_significant_ties(tmp_path):
    """Ensures CD diagrams cleanly handle the edge case where all arms differ from the baseline."""
    mr = pd.Series({
        "flash": 1.0, "linear": 2.0, "linformer": 3.0, "vanilla": 4.0, "sparse": 5.0
    })
    
    # Fabricate a post-hoc where everything is significant
    ph = posthoc.compare_to_control(mr, "flash", 40, "m", method="finner")
    out = tmp_path / "cd_all_sig.png"
    
    plots.critical_difference_diagram(mr, ph, "all significant", out)
    assert out.exists()
    assert out.stat().st_size > 1000