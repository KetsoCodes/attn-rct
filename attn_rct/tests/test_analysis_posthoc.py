"""Unit tests for the post-hoc statistical comparison stage.

Ensures the integrity of p-value adjustments (Holm, Finner), proper 
comparison counting (control vs. all-pairs), and correct graph clique 
generation for visualizations. Crucially validates that adjusted p-values 
remain monotonic and never falsely inflate significance.
"""

import numpy as np
import pandas as pd
import pytest

from attn_rct.analysis import posthoc

RANKS = pd.Series({"flash": 1.0, "linear": 2.0, "linformer": 3.0,
                   "vanilla": 4.0, "sparse": 5.0})


def test_standard_error_formula():
    """Verifies the standard error formula SE = sqrt(k(k+1) / 6N)."""
    assert posthoc.standard_error(6, 5) == pytest.approx(np.sqrt(5 * 6 / 36.0))


def test_control_family_has_k_minus_one_rows():
    """Ensures the control family compares every variant except the baseline against the baseline."""
    result = posthoc.compare_to_control(RANKS, "flash", 6, "m")
    assert len(result.table) == len(RANKS) - 1
    assert "flash" not in set(result.table["variant"])


def test_all_pairs_has_k_choose_two_rows():
    """Ensures the all-pairs family covers every unordered pair."""
    result = posthoc.all_pairs_nemenyi(RANKS, 6, "m")
    k = len(RANKS)
    assert len(result.table) == k * (k - 1) // 2


def test_holm_is_monotone_and_never_shrinks_p():
    """Verifies Holm adjusted p-values are non-decreasing and bounded by the raw p-values."""
    raw = np.array([0.001, 0.04, 0.03, 0.20, 0.6])
    adjusted = posthoc.holm_adjust(raw)
    assert np.all(adjusted >= raw - 1e-12)
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= -1e-12)
    assert np.all(adjusted <= 1.0)


def test_finner_is_monotone_and_never_shrinks_p():
    """Verifies Finner adjusted p-values are non-decreasing and bounded by the raw p-values."""
    raw = np.array([0.001, 0.04, 0.03, 0.20, 0.6])
    adjusted = posthoc.finner_adjust(raw)
    assert np.all(adjusted >= raw - 1e-12)
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= -1e-12)
    assert np.all(adjusted <= 1.0)


def test_finner_is_no_more_conservative_than_holm():
    """Ensures Finner adjustment is at least as powerful (produces equal or smaller p) as Holm."""
    raw = np.array([0.001, 0.01, 0.03, 0.20])
    assert np.all(posthoc.finner_adjust(raw) <= posthoc.holm_adjust(raw) + 1e-12)


def test_holm_smallest_p_multiplied_by_m():
    """Checks that the most extreme comparison is corrected by the full family size."""
    raw = np.array([0.01, 0.5, 0.6, 0.7])
    assert posthoc.holm_adjust(raw)[0] == pytest.approx(4 * 0.01)


def test_better_than_baseline_flag_matches_rank_direction():
    """Verifies that a lower numerical rank correctly flags as 'better than baseline'."""
    result = posthoc.compare_to_control(RANKS, "linformer", 6, "m")
    for _, row in result.table.iterrows():
        assert row["better_than_baseline"] == (row["mean_rank"] < row["baseline_rank"])


def test_control_family_is_more_powerful_than_all_pairs():
    """Ensures correcting for k-1 comparisons detects equal or more effects than all-pairs."""
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
    """Checks that increasing sample size (N) correctly tightens the critical difference."""
    assert posthoc.critical_difference(5, 8) < posthoc.critical_difference(5, 6)
    assert posthoc.critical_difference(5, 20) < posthoc.critical_difference(5, 8)


def test_critical_difference_known_value():
    """Verifies the calculated CD matches the exact tabulated Nemenyi value for k=5, N=8."""
    assert posthoc.critical_difference(5, 8) == pytest.approx(2.157, abs=1e-3)


def test_unknown_baseline_is_refused():
    """Ensures passing a non-existent baseline raises an error."""
    with pytest.raises(ValueError, match="not among variants"):
        posthoc.compare_to_control(RANKS, "nonexistent", 6, "m")


def test_unknown_method_is_refused():
    """Ensures unsupported adjustment methods raise an error."""
    with pytest.raises(ValueError, match="unknown method"):
        posthoc.compare_to_control(RANKS, "flash", 6, "m", method="bonferroni_ish")


def test_untabulated_k_is_refused():
    """Ensures 'k' values lacking tabulated critical values raise an error instead of interpolating."""
    big = pd.Series({f"v{i}": float(i) for i in range(1, 13)})
    with pytest.raises(ValueError, match="no tabulated q"):
        posthoc.all_pairs_nemenyi(big, 8, "m")


def test_identical_ranks_produce_no_significance():
    """Verifies perfectly tied ranks yield no significant differences."""
    tied = pd.Series({v: 3.0 for v in RANKS.index})
    result = posthoc.compare_to_control(tied, "flash", 8, "m")
    assert not result.table["reject"].any()


def test_cliques_group_indistinguishable_arms():
    """Checks that all-pairs cliques group arms within the critical difference."""
    result = posthoc.all_pairs_nemenyi(RANKS, 6, "m")
    groups = posthoc.cliques(RANKS, result)
    assert any("flash" in g and "linear" in g for g in groups)


def test_cliques_reflect_control_family_decisions():
    """Ensures control family cliques never group the baseline with significantly different arms."""
    result = posthoc.compare_to_control(RANKS, "flash", 6, "m", method="finner")
    groups = posthoc.cliques(RANKS, result)
    significant = set(result.significant()["variant"])
    
    for group in groups:
        if "flash" in group:
            assert not (set(group) & significant)


def test_control_cliques_do_not_join_untested_arms():
    """Verifies control cliques only group the baseline with its statistical ties, avoiding untested pairs."""
    result = posthoc.compare_to_control(RANKS, "flash", 8, "mem", method="finner")
    groups = posthoc.cliques(RANKS, result)
    
    assert len(groups) <= 1
    for group in groups:
        assert "flash" in group
        non_base = [v for v in group if v != "flash"]
        sig = {r["variant"] for _, r in result.table.iterrows() if r["reject"]}
        assert all(v not in sig for v in non_base)


def test_control_clique_is_baseline_plus_its_ties():
    """Ensures the control group exactly equals the baseline and its non-significant variants."""
    result = posthoc.compare_to_control(RANKS, "flash", 8, "mem", method="finner")
    sig = {r["variant"] for _, r in result.table.iterrows() if r["reject"]}
    ties = {v for v in RANKS.index if v != "flash" and v not in sig}
    groups = posthoc.cliques(RANKS, result)
    
    if ties:
        assert len(groups) == 1
        assert set(groups[0]) == ties | {"flash"}
    else:
        assert groups == []