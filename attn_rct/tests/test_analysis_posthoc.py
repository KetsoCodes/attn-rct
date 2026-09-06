"""Unit tests for post-hoc statistical comparisons.

Ensures that step-down p-value adjustments (Holm, Finner) maintain strict monotonicity, 
do not artificially inflate significance, and that control vs. all-pairs families 
generate the mathematically correct number of comparisons.
"""

import numpy as np
import pandas as pd
import pytest

from attn_rct.analysis import posthoc

RANKS = pd.Series({"flash": 1.0, "linear": 2.0, "linformer": 3.0,
                   "vanilla": 4.0, "sparse": 5.0})


def test_standard_error_formula():
    """Verifies standard error computes correctly as sqrt(k(k+1) / 6N)."""
    assert posthoc.standard_error(6, 5) == pytest.approx(np.sqrt(5 * 6 / 36.0))


def test_control_family_has_k_minus_one_rows():
    """Checks that the against-control method strictly executes (k-1) comparisons."""
    result = posthoc.compare_to_control(RANKS, "flash", 6, "m")
    assert len(result.table) == len(RANKS) - 1
    assert "flash" not in set(result.table["variant"])


def test_all_pairs_has_k_choose_two_rows():
    """Checks that the Nemenyi all-pairs method correctly executes k(k-1)/2 comparisons."""
    result = posthoc.all_pairs_nemenyi(RANKS, 6, "m")
    k = len(RANKS)
    assert len(result.table) == k * (k - 1) // 2


def test_holm_is_monotone_and_never_shrinks_p():
    """Ensures Holm adjustments are strictly non-decreasing and do not falsely shrink p-values."""
    raw = np.array([0.001, 0.04, 0.03, 0.20, 0.6])
    adjusted = posthoc.holm_adjust(raw)
    
    assert np.all(adjusted >= raw - 1e-12)
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= -1e-12)
    assert np.all(adjusted <= 1.0)


def test_finner_is_monotone_and_never_shrinks_p():
    """Ensures Finner adjustments are strictly non-decreasing and do not falsely shrink p-values."""
    raw = np.array([0.001, 0.04, 0.03, 0.20, 0.6])
    adjusted = posthoc.finner_adjust(raw)
    
    assert np.all(adjusted >= raw - 1e-12)
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= -1e-12)
    assert np.all(adjusted <= 1.0)


def test_finner_is_no_more_conservative_than_holm():
    """Validates that Finner yields equal or lower adjusted p-values compared to Holm."""
    raw = np.array([0.001, 0.01, 0.03, 0.20])
    assert np.all(posthoc.finner_adjust(raw) <= posthoc.holm_adjust(raw) + 1e-12)


def test_holm_smallest_p_multiplied_by_m():
    """Checks that the most extreme comparison in Holm is corrected by the full family size."""
    raw = np.array([0.01, 0.5, 0.6, 0.7])
    assert posthoc.holm_adjust(raw)[0] == pytest.approx(4 * 0.01)


def test_better_than_baseline_flag_matches_rank_direction():
    """Ensures the boolean flag for 'better_than_baseline' matches the rank comparison logic."""
    result = posthoc.compare_to_control(RANKS, "linformer", 6, "m")
    for _, row in result.table.iterrows():
        assert row["better_than_baseline"] == (row["mean_rank"] < row["baseline_rank"])


def test_control_family_is_more_powerful_than_all_pairs():
    """Verifies that the against-control method detects equal or more differences than all-pairs."""
    control = posthoc.compare_to_control(RANKS, "flash", 6, "m", method="finner")
    pairs = posthoc.all_pairs_nemenyi(RANKS, 6, "m")
    
    control_hits = set(control.significant()["variant"])
    pair_hits = {
        row["variant_b"] for _, row in pairs.significant().iterrows() if row["variant_a"] == "flash"
    } | {
        row["variant_a"] for _, row in pairs.significant().iterrows() if row["variant_b"] == "flash"
    }
    
    assert control_hits >= pair_hits


def test_critical_difference_shrinks_as_n_grows():
    """Checks that statistical power improves (difference shrinks) as sample size (N) increases."""
    assert posthoc.critical_difference(5, 8) < posthoc.critical_difference(5, 6)
    assert posthoc.critical_difference(5, 20) < posthoc.critical_difference(5, 8)


def test_critical_difference_known_value():
    """Validates the exact mathematical output for the project's target design space (k=5, N=8)."""
    assert posthoc.critical_difference(5, 8) == pytest.approx(2.157, abs=1e-3)


def test_unknown_baseline_is_refused():
    """Ensures the test rejects execution if an unrecognized baseline string is passed."""
    with pytest.raises(ValueError, match="not among variants"):
        posthoc.compare_to_control(RANKS, "nonexistent", 6, "m")


def test_unknown_method_is_refused():
    """Ensures the test rejects execution if an unrecognized adjustment method is passed."""
    with pytest.raises(ValueError, match="unknown method"):
        posthoc.compare_to_control(RANKS, "flash", 6, "m", method="bonferroni_ish")


def test_untabulated_k_is_refused():
    """Checks that Nemenyi strictly fails if 'k' is outside standard tabulated critical values."""
    big = pd.Series({f"v{i}": float(i) for i in range(1, 13)})
    with pytest.raises(ValueError, match="no tabulated q"):
        posthoc.all_pairs_nemenyi(big, 8, "m")


def test_identical_ranks_produce_no_significance():
    """Verifies completely tied ranks appropriately result in no statistical significance."""
    tied = pd.Series({v: 3.0 for v in RANKS.index})
    result = posthoc.compare_to_control(tied, "flash", 8, "m")
    assert not result.table["reject"].any()


def test_cliques_group_indistinguishable_arms():
    """Checks that variants with differences below the Nemenyi threshold are grouped together."""
    result = posthoc.all_pairs_nemenyi(RANKS, 6, "m")
    groups = posthoc.cliques(RANKS, result)
    assert any("flash" in g and "linear" in g for g in groups)


def test_cliques_reflect_control_family_decisions():
    """Ensures clique grouping for control-family tests relies directly on adjusted p-values."""
    result = posthoc.compare_to_control(RANKS, "flash", 6, "m", method="finner")
    groups = posthoc.cliques(RANKS, result)
    significant = set(result.significant()["variant"])
    
    for group in groups:
        if "flash" in group:
            assert not (set(group) & significant)