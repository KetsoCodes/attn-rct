"""Post-hoc statistical comparisons to identify which specific variants differ."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PostHocResult:
    """Results from adjusted post-hoc comparisons for a single metric."""

    metric: str
    family: str
    baseline: str | None
    n_blocks: int
    n_variants: int
    alpha: float
    method: str
    table: pd.DataFrame

    def significant(self) -> pd.DataFrame:
        """Filters the results to return only statistically significant comparisons."""
        return self.table[self.table["reject"]]

    def summary(self) -> str:
        """Generates a formatted text summary of the comparison results and significance."""
        head = (f"metric   : {self.metric}\n"
                f"family   : {self.family}"
                + (f" (baseline = {self.baseline})" if self.baseline else "")
                + f"\nmethod   : {self.method}   alpha={self.alpha}   "
                  f"N={self.n_blocks} k={self.n_variants}")
        body = self.table.to_string(index=False,
                                    float_format=lambda v: f"{v:.4f}")
        n_sig = int(self.table["reject"].sum())
        return f"{head}\n{body}\n{n_sig} of {len(self.table)} comparisons significant"


def standard_error(n_blocks: int, k: int) -> float:
    """Calculates the standard error of a mean-rank difference under the null hypothesis."""
    return float(np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def _z_and_p(diff: float, se: float) -> tuple[float, float]:
    """Computes the two-tailed z-statistic and unadjusted p-value for a given difference."""
    z = diff / se
    return z, float(2.0 * stats.norm.sf(abs(z)))


def holm_adjust(p_values) -> np.ndarray:
    """Applies the Holm step-down method to adjust p-values and enforces monotonicity."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


def finner_adjust(p_values) -> np.ndarray:
    """Applies the Finner step-down method to adjust p-values, controlling family-wise error rate."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        i = rank + 1
        value = 1.0 - (1.0 - p[idx]) ** (m / i)
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


ADJUSTERS = {"holm": holm_adjust, "finner": finner_adjust}


def compare_to_control(mean_ranks: pd.Series, baseline: str, n_blocks: int,
                       metric: str, method="finner", alpha=0.05) -> PostHocResult:
    """Compares every variant to a designated baseline control, adjusting for multiple comparisons."""
    if baseline not in mean_ranks.index:
        raise ValueError(
            f"baseline {baseline!r} not among variants {sorted(mean_ranks.index)}"
        )
    if method not in ADJUSTERS:
        raise ValueError(f"unknown method {method!r}; expected {sorted(ADJUSTERS)}")

    k = len(mean_ranks)
    se = standard_error(n_blocks, k)
    base_rank = float(mean_ranks[baseline])

    rows = []
    for variant, rank in mean_ranks.items():
        if variant == baseline:
            continue
        diff = float(rank) - base_rank
        z, p = _z_and_p(diff, se)
        rows.append({
            "variant": variant, "baseline": baseline,
            "mean_rank": float(rank), "baseline_rank": base_rank,
            "rank_diff": diff, "z": z, "p_unadjusted": p,
        })

    table = pd.DataFrame(rows)
    
    # Extracted function assignment to prevent Markdown link rendering bugs in external editors
    adjust_func = ADJUSTERS[method]
    table["p_adjusted"] = adjust_func(table["p_unadjusted"].to_numpy())
    
    table["reject"] = table["p_adjusted"] < alpha
    table["better_than_baseline"] = table["rank_diff"] < 0
    table = table.sort_values("p_adjusted").reset_index(drop=True)

    return PostHocResult(metric=metric, family="control", baseline=baseline,
                         n_blocks=n_blocks, n_variants=k, alpha=alpha,
                         method=method, table=table)


NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
NEMENYI_Q10 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
               7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920}


def critical_difference(k: int, n_blocks: int, alpha=0.05) -> float:
    """Calculates the Nemenyi critical difference threshold required to deem two variants statistically distinct."""
    table = {0.05: NEMENYI_Q05, 0.10: NEMENYI_Q10}.get(alpha)
    if table is None:
        raise ValueError(f"no tabulated q for alpha={alpha}; use 0.05 or 0.10")
    if k not in table:
        raise ValueError(f"no tabulated q for k={k}; supported k are {sorted(table)}")
    return table[k] * standard_error(n_blocks, k)


def all_pairs_nemenyi(mean_ranks: pd.Series, n_blocks: int, metric: str,
                      alpha=0.05) -> PostHocResult:
    """Compares every pair of variants against the Nemenyi critical difference threshold."""
    k = len(mean_ranks)
    cd = critical_difference(k, n_blocks, alpha)

    rows = []
    for a, b in itertools.combinations(mean_ranks.index, 2):
        diff = float(mean_ranks[a]) - float(mean_ranks[b])
        rows.append({
            "variant_a": a, "variant_b": b,
            "rank_a": float(mean_ranks[a]), "rank_b": float(mean_ranks[b]),
            "abs_rank_diff": abs(diff),
            "critical_difference": cd,
            "reject": abs(diff) > cd,
        })

    table = pd.DataFrame(rows).sort_values("abs_rank_diff", ascending=False)
    table = table.reset_index(drop=True)
    return PostHocResult(metric=metric, family="all_pairs", baseline=None,
                         n_blocks=n_blocks, n_variants=k, alpha=alpha,
                         method="nemenyi", table=table)


def cliques(mean_ranks: pd.Series, result: PostHocResult) -> list[list[str]]:
    """Groups variants into maximal ordered cliques where no significant statistical difference exists internally."""
    ordered = list(mean_ranks.sort_values().index)
    if result.family == "all_pairs":
        cd = float(result.table["critical_difference"].iloc[0])
        def indistinguishable(a, b):
            return abs(float(mean_ranks[a]) - float(mean_ranks[b])) <= cd
    else:
        significant = {
            row["variant"] for _, row in result.table.iterrows() if row["reject"]
        }
        base = result.baseline
        def indistinguishable(a, b):
            if base in (a, b):
                other = b if a == base else a
                return other not in significant
            return True

    groups = []
    for i, first in enumerate(ordered):
        group = [first]
        for other in ordered[i + 1:]:
            if all(indistinguishable(member, other) for member in group):
                group.append(other)
            else:
                break
        if not any(set(group) <= set(existing) for existing in groups):
            groups.append(group)
    return [g for g in groups if len(g) > 1]