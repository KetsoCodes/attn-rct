"""The omnibus gatekeeper: determining if attention variants exhibit statistically significant differences.

This stage strictly controls family-wise error rates. Downstream pairwise comparisons 
are only licensed if this omnibus test rejects the null hypothesis.

The script computes three statistics to handle the specific edge cases of small sample sizes (N=8) 
and perfect rank consistency (which crashes standard parametric approximations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class OmnibusResult:
    """The statistical outcome of the omnibus test for a single metric."""

    metric: str
    n_blocks: int
    n_variants: int
    mean_ranks: dict
    friedman_chi2: float
    friedman_p: float
    iman_davenport_f: float | None
    iman_davenport_p: float | None
    permutation_p: float | None
    n_permutations: int
    alpha: float
    notes: list = field(default_factory=list)

    @property
    def reject(self) -> bool:
        """Determines rejection using the permutation p-value where available, falling back to Friedman."""
        p = self.permutation_p if self.permutation_p is not None else self.friedman_p
        return p < self.alpha

    def summary(self) -> str:
        """Formats the test results for terminal output."""
        order = ", ".join(f"{v} {r:.2f}" for v, r in
                          sorted(self.mean_ranks.items(), key=lambda kv: kv[1]))
        lines = [
            f"metric        : {self.metric}",
            f"blocks (N)    : {self.n_blocks}    variants (k): {self.n_variants}",
            f"mean ranks    : {order}",
            f"Friedman      : chi2={self.friedman_chi2:.4f}  p={self.friedman_p:.5f}",
        ]
        if self.iman_davenport_f is None:
            lines.append("Iman-Davenport: undefined (perfectly consistent ranking)")
        else:
            lines.append(f"Iman-Davenport: F={self.iman_davenport_f:.4f}  p={self.iman_davenport_p:.5f}")
            
        if self.permutation_p is not None:
            lines.append(f"permutation   : p={self.permutation_p:.5f} ({self.n_permutations:,} shuffles)")
            
        lines.append(f"decision      : {'REJECT' if self.reject else 'fail to reject'} H0 at alpha={self.alpha}")
        lines.extend(f"note          : {n}" for n in self.notes)
        return "\n".join(lines)


def friedman_chi2(ranks: np.ndarray) -> float:
    """Calculates the standard Friedman chi-square statistic from a (blocks x variants) rank matrix."""
    n_blocks, k = ranks.shape
    mean_rank = ranks.mean(axis=0)
    return (12.0 * n_blocks / (k * (k + 1))) * (np.sum(mean_rank ** 2) - k * (k + 1) ** 2 / 4.0)


def iman_davenport(chi2: float, n_blocks: int, k: int):
    """Applies the Iman-Davenport correction to the Friedman statistic.

    Returns:
        (F_stat, p_value) or (None, None) if the ranking is perfectly consistent, 
        which drives the denominator to zero.
    """
    denominator = n_blocks * (k - 1) - chi2
    if denominator <= 1e-12:
        return None, None
    f_stat = (n_blocks - 1) * chi2 / denominator
    p = stats.f.sf(f_stat, k - 1, (k - 1) * (n_blocks - 1))
    return f_stat, p


def permutation_p(ranks: np.ndarray, n_permutations=100_000, seed=0) -> float:
    """Calculates an empirical p-value by shuffling variant labels within each block.
    
    This avoids asymptotic approximations and remains mathematically sound even with 
    perfect rank separation. Uses the add-one smoothing format.
    """
    observed = friedman_chi2(ranks)
    rng = np.random.default_rng(seed)
    n_blocks, k = ranks.shape

    count = 0
    for _ in range(n_permutations):
        shuffled = rng.permuted(ranks, axis=1)
        if friedman_chi2(shuffled) >= observed - 1e-12:
            count += 1
            
    return (1.0 + count) / (1.0 + n_permutations)


def run_omnibus(rank_matrix: pd.DataFrame, metric: str, alpha=0.05, n_permutations=20_000, seed=0) -> OmnibusResult:
    """Executes the Friedman, Iman-Davenport, and permutation tests on the ranked data matrix.

    Raises:
        ValueError: If the rank matrix contains NaN values or has fewer than 2 blocks/variants.
    """
    if rank_matrix.isna().any().any():
        raise ValueError("rank matrix contains NaN; the design x variant matrix is incomplete and cannot be tested")
        
    ranks = rank_matrix.to_numpy(dtype=float)
    n_blocks, k = ranks.shape
    
    if n_blocks < 2 or k < 2:
        raise ValueError(f"need at least 2 blocks and 2 variants, got {ranks.shape}")

    chi2 = friedman_chi2(ranks)
    chi2_p = stats.chi2.sf(chi2, k - 1)
    f_stat, f_p = iman_davenport(chi2, n_blocks, k)
    perm_p = permutation_p(ranks, n_permutations=n_permutations, seed=seed)

    notes = []
    if f_stat is None:
        notes.append(
            "Iman-Davenport is undefined here because every block ranks the variants identically; "
            "this is maximal evidence against the null, not a failure. The permutation p-value is the decision of record."
        )
    if n_blocks < 8:
        notes.append(
            f"N={n_blocks} blocks is small; the chi-square approximation is thin and the "
            "permutation p-value should be preferred."
        )

    return OmnibusResult(
        metric=metric,
        n_blocks=n_blocks,
        n_variants=k,
        mean_ranks=dict(zip(rank_matrix.columns, ranks.mean(axis=0))),
        friedman_chi2=chi2,
        friedman_p=chi2_p,
        iman_davenport_f=f_stat,
        iman_davenport_p=f_p,
        permutation_p=perm_p,
        n_permutations=n_permutations,
        alpha=alpha,
        notes=notes,
    )