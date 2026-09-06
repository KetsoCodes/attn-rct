"""Unit tests for the omnibus statistical gatekeeper stage.

Ensures that the Friedman, Iman-Davenport, and permutation tests correctly manage 
Type I errors (false positives), detect valid effects, and safely handle edge cases 
such as perfect rank consistency (which breaks standard asymptotic approximations).
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from attn_rct.analysis import aggregate, collect, omnibus, synth

VARIANTS = ["flash", "linear", "linformer", "vanilla", "sparse"]


def rank_frame(rows):
    """Builds a (blocks x variants) rank DataFrame from a list of rows."""
    return pd.DataFrame(rows, columns=VARIANTS)


def test_friedman_matches_scipy():
    """Verifies the custom Friedman chi-square calculation matches SciPy's implementation."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        data = rng.normal(size=(8, 5))
        ranks = np.apply_along_axis(stats.rankdata, 1, data)
        ours = omnibus.friedman_chi2(ranks)
        theirs = stats.friedmanchisquare(*data.T).statistic
        assert ours == pytest.approx(theirs, abs=1e-9)


def test_null_case_does_not_reject(tmp_path):
    """Ensures the tests do not produce false positives (Type I errors) on random noise."""
    synth.generate(tmp_path, effect=0.0, seed=7)
    df, report = collect.load_results(tmp_path, phase="full")
    cells = aggregate.aggregate_seeds(collect.require_complete(df, report))
    ranked = aggregate.rank_within_blocks(cells, "best_val_accuracy")
    result = omnibus.run_omnibus(aggregate.rank_matrix(ranked), "accuracy",
                                 n_permutations=2000)
    
    assert not result.reject
    assert result.friedman_p > 0.05
    assert result.permutation_p > 0.05


def test_planted_effect_rejects(tmp_path):
    """Verifies that a known synthetic effect is successfully detected."""
    synth.generate(tmp_path, effect=1.0, seed=0)
    df, report = collect.load_results(tmp_path, phase="full")
    cells = aggregate.aggregate_seeds(collect.require_complete(df, report))
    ranked = aggregate.rank_within_blocks(cells, "mean_train_seconds_per_epoch")
    result = omnibus.run_omnibus(aggregate.rank_matrix(ranked), "sec/epoch",
                                 n_permutations=2000)
    
    assert result.reject
    assert result.permutation_p < 0.01


def test_perfect_separation_leaves_iman_davenport_undefined():
    """Checks that perfect rank consistency safely reports the Iman-Davenport statistic as undefined.
    
    This verifies the pipeline will not crash or emit misleading statistics when evaluating metrics 
    with perfect separation (like the project's memory metric).
    """
    matrix = rank_frame([[1., 2., 3., 4., 5.]] * 5)
    result = omnibus.run_omnibus(matrix, "peak_memory_mb", n_permutations=2000)
    
    assert result.iman_davenport_f is None
    assert result.iman_davenport_p is None
    assert result.permutation_p < 0.01
    assert result.reject
    assert any("undefined" in n for n in result.notes)


def test_all_tied_does_not_reject():
    """Ensures the test correctly handles completely tied ranks without falsely rejecting the null."""
    matrix = rank_frame([[3., 3., 3., 3., 3.]] * 6)
    result = omnibus.run_omnibus(matrix, "tied", n_permutations=1000)
    
    assert result.friedman_chi2 == pytest.approx(0.0, abs=1e-9)
    assert not result.reject


def test_permutation_p_is_never_zero():
    """Verifies the empirical p-value remains strictly positive using the add-one smoothing method."""
    matrix = rank_frame([[1., 2., 3., 4., 5.]] * 8)
    p = omnibus.permutation_p(matrix.to_numpy(), n_permutations=500, seed=0)
    
    assert p > 0
    assert p == pytest.approx(1.0 / 501, abs=1e-9)


def test_permutation_is_reproducible():
    """Ensures the permutation test yields deterministic results when provided a fixed seed."""
    matrix = rank_frame([[1., 3., 2., 5., 4.], [2., 1., 3., 4., 5.],
                         [1., 2., 4., 3., 5.], [1., 3., 2., 4., 5.]])
    a = omnibus.permutation_p(matrix.to_numpy(), n_permutations=1000, seed=11)
    b = omnibus.permutation_p(matrix.to_numpy(), n_permutations=1000, seed=11)
    
    assert a == b


def test_permutation_agrees_with_chi2_at_larger_n():
    """Checks that the empirical permutation test and the asymptotic Friedman test align at larger sample sizes."""
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(30):
        base = np.array([1., 2., 3., 4., 5.])
        if rng.random() < 0.3:
            i, j = rng.choice(5, 2, replace=False)
            base[[i, j]] = base[[j, i]]
        rows.append(base)
    matrix = rank_frame(rows)
    result = omnibus.run_omnibus(matrix, "consistent", n_permutations=5000)
    
    assert result.permutation_p < 0.01 and result.friedman_p < 0.01


def test_incomplete_matrix_is_refused():
    """Ensures testing is rejected if the input matrix contains missing values (NaN)."""
    matrix = rank_frame([[1., 2., 3., 4., 5.], [1., 2., np.nan, 4., 5.]])
    with pytest.raises(ValueError, match="incomplete"):
        omnibus.run_omnibus(matrix, "broken", n_permutations=100)


def test_too_few_blocks_is_refused():
    """Verifies that the omnibus tests require a minimum of two blocks to execute."""
    matrix = rank_frame([[1., 2., 3., 4., 5.]])
    with pytest.raises(ValueError, match="at least 2 blocks"):
        omnibus.run_omnibus(matrix, "single", n_permutations=100)


def test_small_n_is_flagged():
    """Checks that a warning note is appended when the sample size (N) is too small for reliable asymptotic approximation."""
    matrix = rank_frame([[1., 2., 3., 4., 5.], [2., 1., 3., 5., 4.],
                         [1., 3., 2., 4., 5.]])
    result = omnibus.run_omnibus(matrix, "small", n_permutations=1000)
    
    assert any("N=3" in n for n in result.notes)


def test_reject_prefers_the_permutation_p():
    """Ensures the final rejection decision relies on the exact permutation p-value where available."""
    matrix = rank_frame([[1., 2., 3., 4., 5.]] * 6)
    result = omnibus.run_omnibus(matrix, "perfect", n_permutations=2000)
    
    assert result.permutation_p is not None
    assert result.reject is (result.permutation_p < result.alpha)