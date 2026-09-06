"""Tests for the effect size module, ensuring proper calculation of magnitudes, confidence intervals, and metric directions."""

import numpy as np
import pandas as pd
import pytest

from attn_rct.analysis import aggregate, collect, effects, synth


def matrix(**columns):
    """Helper function to build a blocks-by-variants value matrix."""
    return pd.DataFrame(columns)


def test_cohens_dz_is_mean_over_sd_of_differences():
    """Verifies that paired Cohen's d_z divides by the standard deviation of the differences."""
    differences = np.array([1.0, 2.0, 3.0, 4.0])
    expected = differences.mean() / differences.std(ddof=1)
    assert effects.cohens_dz(differences) == pytest.approx(expected)


def test_cohens_dz_zero_when_no_difference():
    """Ensures Cohen's d_z is exactly zero when all differences are zero."""
    assert effects.cohens_dz(np.zeros(6)) == 0.0


def test_cohens_dz_infinite_when_perfectly_consistent():
    """Verifies Cohen's d_z handles zero-variance constant differences by returning infinity."""
    assert np.isinf(effects.cohens_dz(np.full(5, 2.0)))


def test_geometric_mean_ratio_is_symmetric_under_inversion():
    """Verifies that the geometric mean correctly handles inverse ratios symmetrically."""
    base = np.array([10.0, 10.0])
    variant = np.array([5.0, 20.0])
    assert effects.geometric_mean_ratio(base, variant) == pytest.approx(1.0)


def test_geometric_mean_ratio_reads_as_a_speedup():
    """Ensures the geometric ratio accurately reflects a baseline-to-variant speedup or efficiency gain."""
    base = np.array([5.0, 5.0, 5.0])
    variant = np.array([10.0, 10.0, 10.0])
    assert effects.geometric_mean_ratio(base, variant) == pytest.approx(0.5)


def test_geometric_mean_ratio_nan_on_nonpositive():
    """Ensures the geometric ratio returns NaN when encountering zero or negative values."""
    assert np.isnan(effects.geometric_mean_ratio(np.array([1.0, 0.0]),
                                                 np.array([1.0, 1.0])))


def test_bootstrap_ci_brackets_the_point_estimate():
    """Verifies that the computed bootstrap confidence interval contains the point estimate."""
    rng = np.random.default_rng(0)
    differences = rng.normal(2.0, 1.0, size=20)
    point = effects.cohens_dz(differences)
    low, high = effects.bootstrap_ci(differences, effects.cohens_dz,
                                     n_bootstrap=2000, seed=1)
    assert low < point < high


def test_bootstrap_ci_is_reproducible():
    """Ensures the bootstrap confidence interval yields identical results given the same random seed."""
    differences = np.array([0.5, 1.2, -0.3, 0.8, 1.1, 0.2])
    a = effects.bootstrap_ci(differences, effects.cohens_dz, n_bootstrap=1000, seed=5)
    b = effects.bootstrap_ci(differences, effects.cohens_dz, n_bootstrap=1000, seed=5)
    assert a == b


def test_bootstrap_ci_widens_at_smaller_n():
    """Verifies that the confidence interval correctly widens when the number of blocks decreases."""
    rng = np.random.default_rng(2)
    large = rng.normal(1.0, 1.0, size=40)
    small = large[:5]
    wide = effects.bootstrap_ci(small, effects.cohens_dz, n_bootstrap=2000, seed=0)
    narrow = effects.bootstrap_ci(large, effects.cohens_dz, n_bootstrap=2000, seed=0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_favours_respects_higher_is_better():
    """Ensures the 'favours' label correctly identifies the superior variant for a higher-is-better metric."""
    values = matrix(flash=[0.30, 0.31, 0.32], linear=[0.35, 0.36, 0.37])
    result = effects.compute_effects(values, "flash", "acc", "higher", n_bootstrap=500)
    assert result.table.loc[0, "favours"] == "linear"


def test_favours_respects_lower_is_better():
    """Ensures the 'favours' label correctly identifies the superior variant for a lower-is-better metric."""
    values = matrix(flash=[100.0, 110.0, 120.0], sparse=[400.0, 410.0, 420.0])
    result = effects.compute_effects(values, "flash", "mem", "lower", n_bootstrap=500)
    assert result.table.loc[0, "favours"] == "flash"


def test_direction_flip_reverses_favours():
    """Verifies that flipping the metric direction perfectly reverses which variant is favoured."""
    values = matrix(flash=[1.0, 2.0, 3.0], other=[4.0, 5.0, 6.0])
    higher = effects.compute_effects(values, "flash", "m", "higher", n_bootstrap=300)
    lower = effects.compute_effects(values, "flash", "m", "lower", n_bootstrap=300)
    assert higher.table.loc[0, "favours"] == "other"
    assert lower.table.loc[0, "favours"] == "flash"


def test_ratio_only_for_lower_is_better_by_default():
    """Ensures geometric ratios are exclusively computed for lower-is-better efficiency metrics by default."""
    values = matrix(flash=[1.0, 2.0], other=[2.0, 4.0])
    assert "geometric_ratio" in effects.compute_effects(
        values, "flash", "m", "lower", n_bootstrap=200).table.columns
    assert "geometric_ratio" not in effects.compute_effects(
        values, "flash", "m", "higher", n_bootstrap=200).table.columns


def test_identical_arms_give_negligible_effect():
    """Verifies that arms computing the identical underlying function do not produce a statistically meaningful effect."""
    rng = np.random.default_rng(7)
    base = rng.normal(0.35, 0.01, size=8)
    values = matrix(flash=base, vanilla=base + rng.normal(0, 0.001, size=8))
    result = effects.compute_effects(values, "flash", "acc", "higher", n_bootstrap=2000)
    row = result.table.iloc[0]
    assert abs(row["cohens_dz"]) < 0.8
    assert not row["ci_excludes_zero"]


def test_missing_baseline_is_refused():
    """Ensures attempting to compute effects with an invalid baseline throws an explicit error."""
    values = matrix(a=[1.0, 2.0], b=[3.0, 4.0])
    with pytest.raises(ValueError, match="not among variants"):
        effects.compute_effects(values, "flash", "m", "lower", n_bootstrap=100)


def test_nan_matrix_is_refused():
    """Ensures attempting to compute effects on an incomplete value matrix containing NaNs throws an explicit error."""
    values = matrix(flash=[1.0, np.nan], other=[3.0, 4.0])
    with pytest.raises(ValueError, match="incomplete"):
        effects.compute_effects(values, "flash", "m", "lower", n_bootstrap=100)


def test_too_few_blocks_is_refused():
    """Ensures computing effects with fewer than two blocks throws an explicit error."""
    values = matrix(flash=[1.0], other=[2.0])
    with pytest.raises(ValueError, match="at least 2 blocks"):
        effects.compute_effects(values, "flash", "m", "lower", n_bootstrap=100)


def test_bad_direction_is_refused():
    """Ensures providing an invalid metric direction throws an explicit error."""
    values = matrix(flash=[1.0, 2.0], other=[3.0, 4.0])
    with pytest.raises(ValueError, match="direction must be"):
        effects.compute_effects(values, "flash", "m", "sideways", n_bootstrap=100)


def test_end_to_end_on_synthetic_grid(tmp_path):
    """Verifies the effect size computation runs successfully end-to-end on a full synthetic grid."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    df, report = collect.load_results(tmp_path, phase="full")
    cells = aggregate.aggregate_seeds(collect.require_complete(df, report))
    values = aggregate.value_matrix(cells, "peak_memory_mb")
    result = effects.compute_effects(values, "flash", "peak_memory_mb", "lower",
                                     n_bootstrap=1000)
    assert len(result.table) == 4
    assert (result.table["favours"] == "flash").all()
    assert (result.table["geometric_ratio"] < 1.0).all()