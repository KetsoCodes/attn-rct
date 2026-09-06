"""Tests for the post-hoc comparison module, ensuring statistical adjustments are valid and properly sized."""

import numpy as np
import pandas as pd
import pytest

from attn_rct.analysis import posthoc

RANKS = pd.Series({"flash": 1.0, "linear": 2.0, "linformer": 3.0,
                   "vanilla": 4.0, "sparse": 5.0})


def test_standard_error_formula():
    """Verifies the standard error calculation for mean-rank differences."""
    assert posthoc.standard_error(6, 5) == pytest.approx(np.sqrt(5 * 6 / 36.0))


def test_control_family_has_k_minus_one_rows():
    """Verifies the control family tests all variants exactly once against the baseline."""
    result = posthoc.compare_to_control(RANKS, "flash", 6, "m")
    assert len(result.table) == len(RANKS) - 1
    assert "flash" not in set(result.table["variant"])


def test_all_pairs_has_k_choose_two_rows():
    """Verifies the all-pairs family generates comparisons for every unique pair of variants."""
    result = posthoc.all_pairs_nemenyi(RANKS, 6, "m")
    k = len(RANKS)
    assert len(result.table) == k * (k - 1) // 2


def test_holm_is_monotone_and_never_shrinks_p():
    """Ensures Holm adjustments strictly increase or maintain p-values and preserve monotonic ordering."""
    raw = np.array([0.001, 0.04, 0.03, 0.20, 0.6])
    adjusted = posthoc.holm_adjust(raw)
    assert np.all(adjusted >= raw - 1e-12)
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= -1e-12)
    assert np.all(adjusted <= 1.0)


def test_finner_is_monotone_and_never_shrinks_p():
    """Ensures Finner adjustments strictly increase or maintain p-values and preserve monotonic ordering."""
    raw = np.array([0.001, 0.04, 0.03, 0.20, 0.6])
    adjusted = posthoc.finner_adjust(raw)
    assert np.all(adjusted >= raw - 1e-12)
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= -1e-12)
    assert np.all(adjusted <= 1.0)


def test_finner_is_no_more_conservative_than_holm():
    """Verifies the Finner adjustment yields smaller or equal adjusted p-values compared to Holm."""
    raw = np.array([0.001, 0.01, 0.03, 0.20])
    assert np.all(posthoc.finner_adjust(raw) <= posthoc.holm_adjust(raw) + 1e-12)


def test_holm_smallest_p_multiplied_by_m():
    """Verifies the most significant unadjusted comparison is scaled by the total number of tests."""
    raw = np.array([0.01, 0.5, 0.6, 0.7])
    assert posthoc.holm_adjust(raw)[0] == pytest.approx(4 * 0.01)


def test_better_than_baseline_flag_matches_rank_direction():
    """Ensures the boolean flag for better performance correctly corresponds to lower rank values."""
    result = posthoc.compare_to_control(RANKS, "linformer", 6, "m")
    for _, row in result.table.iterrows():
        assert row["better_than_baseline"] == (row["mean_rank"] < row["baseline_rank"])


def test_control_family_is_more_powerful_than_all_pairs():
    """Verifies the control family approach detects at least as many significant differences as the all-pairs approach."""
    control = posthoc.compare_to_control(RANKS, "flash", 6, "m", method="finner")
    pairs = posthoc.all_pairs_nemenyi(RANKS, 6, "m")
    control_hits = set(control.significant()["variant"])
    pair_hits = {
        row["variant_b"] for _, row in pairs.significant().iterrows()
        if row["variant_a"] == "flash"
    } | {
        row["variant_a"] for _, row in pairs.significant().iterrows()
        if row["variant_b"] == "flash"
    }
    assert control_hits >= pair_hits


def test_critical_difference_shrinks_as_n_grows():
    """Verifies that increasing the sample size tightens the critical difference threshold."""
    assert posthoc.critical_difference(5, 8) < posthoc.critical_difference(5, 6)
    assert posthoc.critical_difference(5, 20) < posthoc.critical_difference(5, 8)


def test_critical_difference_known_value():
    """Verifies the critical difference calculation against a known tabulated expectation."""
    assert posthoc.critical_difference(5, 8) == pytest.approx(2.157, abs=1e-3)


def test_unknown_baseline_is_refused():
    """Ensures attempting to use an invalid baseline control throws an explicit error."""
    with pytest.raises(ValueError, match="not among variants"):
        posthoc.compare_to_control(RANKS, "nonexistent", 6, "m")


def test_unknown_method_is_refused():
    """Ensures providing an invalid adjustment method throws an explicit error."""
    with pytest.raises(ValueError, match="unknown method"):
        posthoc.compare_to_control(RANKS, "flash", 6, "m", method="bonferroni_ish")


def test_untabulated_k_is_refused():
    """Ensures providing an unsupported number of variants raises an explicit error instead of interpolating."""
    big = pd.Series({f"v{i}": float(i) for i in range(1, 13)})
    with pytest.raises(ValueError, match="no tabulated q"):
        posthoc.all_pairs_nemenyi(big, 8, "m")


def test_identical_ranks_produce_no_significance():
    """Verifies perfectly tied ranks result in zero statistically significant findings."""
    tied = pd.Series({v: 3.0 for v in RANKS.index})
    result = posthoc.compare_to_control(tied, "flash", 8, "m")
    assert not result.table["reject"].any()


def test_cliques_group_indistinguishable_arms():
    """Verifies variants grouped into cliques cannot be statistically distinguished from one another."""
    result = posthoc.all_pairs_nemenyi(RANKS, 6, "m")
    groups = posthoc.cliques(RANKS, result)
    assert any("flash" in g and "linear" in g for g in groups)


def test_cliques_reflect_control_family_decisions():
    """Verifies clique groupings accurately reflect the specific outcome of control family adjusted tests."""
    result = posthoc.compare_to_control(RANKS, "flash", 6, "m", method="finner")
    groups = posthoc.cliques(RANKS, result)
    significant = set(result.significant()["variant"])
    for group in groups:
        if "flash" in group:
            assert not (set(group) & significant)