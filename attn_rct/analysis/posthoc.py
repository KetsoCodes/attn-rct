"""Post-hoc statistical comparisons to determine which variants differ.

Computes adjusted p-values (Holm, Finner) against a designated control baseline, 
or evaluates all-pairs critical differences (Nemenyi) following a significant omnibus test.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PostHocResult:
    """Adjusted comparisons for one metric."""

    metric: str
    family: str            # "control" or "all_pairs"
    baseline: str | None
    n_blocks: int
    n_variants: int
    alpha: float
    method: str            # "holm", "finner", or "nemenyi"
    table: pd.DataFrame    # one row per comparison

    def significant(self) -> pd.DataFrame:
        """Only the comparisons that survive correction."""
        return self.table[self.table["reject"]]

    def summary(self) -> str:
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
    """Standard error of a mean-rank difference under the null: sqrt(k(k+1) / 6N)."""
    return float(np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def _z_and_p(diff: float, se: float) -> tuple[float, float]:
    """Two-tailed z statistic and unadjusted p for a mean-rank difference."""
    z = diff / se
    return z, float(2.0 * stats.norm.sf(abs(z)))


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values, enforced monotone non-decreasing.

    Sort ascending, multiply the i-th by (m - i), then take a running maximum so the
    adjusted sequence never decreases -- without that the adjusted values could imply a
    less extreme comparison is more significant than a more extreme one.
    """
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
    """Finner step-down adjusted p-values: 1 - (1 - p_(i))^(m/i), made monotone.

    Slightly more powerful than Holm while still controlling the family-wise error rate.
    """
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
    """Compare every variant to one baseline, correcting for k-1 comparisons.

    Args:
        mean_ranks: mean rank per variant, from aggregate.mean_ranks.
        baseline: the control arm every other arm is compared against.
        n_blocks: N, the number of blocks the ranks were computed over.
        metric: metric name, for reporting.
        method: "holm" or "finner".
        alpha: family-wise significance level.

    Raises:
        ValueError: if the baseline is not among the variants, or the method is unknown.
    """
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
    table["p_adjusted"] = ADJUSTERS[method](table["p_unadjusted"].to_numpy())
    table["reject"] = table["p_adjusted"] < alpha
    # Negative rank_diff means the variant ranks BETTER than the baseline, since rank 1
    # is best. Spelling it out avoids a sign error being read as a direction claim.
    table["better_than_baseline"] = table["rank_diff"] < 0
    table = table.sort_values("p_adjusted").reset_index(drop=True)

    return PostHocResult(metric=metric, family="control", baseline=baseline,
                         n_blocks=n_blocks, n_variants=k, alpha=alpha,
                         method=method, table=table)


# Studentised range statistic q_alpha at alpha=0.05, divided by sqrt(2), indexed by k.
# These are the standard Nemenyi critical values tabulated in Demsar (2006) Table 5.
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
NEMENYI_Q10 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
               7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920}


def critical_difference(k: int, n_blocks: int, alpha=0.05) -> float:
    """Nemenyi critical difference: q_alpha * sqrt(k(k+1) / 6N).

    Two arms whose mean ranks differ by less than this are indistinguishable under the
    all-pairs family. This is the bar a critical-difference diagram draws.

    Raises:
        ValueError: for a k with no tabulated critical value, or an unsupported alpha.
    """
    table = {0.05: NEMENYI_Q05, 0.10: NEMENYI_Q10}.get(alpha)
    if table is None:
        raise ValueError(f"no tabulated q for alpha={alpha}; use 0.05 or 0.10")
    if k not in table:
        raise ValueError(f"no tabulated q for k={k}; supported k are {sorted(table)}")
    return table[k] * standard_error(n_blocks, k)


def all_pairs_nemenyi(mean_ranks: pd.Series, n_blocks: int, metric: str,
                      alpha=0.05) -> PostHocResult:
    """Compare every pair of variants using the Nemenyi critical difference.

    Unlike the control family this uses a single critical difference rather than adjusted
    p-values, which is what makes it the family a classic CD diagram depicts.
    """
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
    """Group variants that the post-hoc cannot distinguish, for a CD diagram.

    A classic CD diagram joins arms with a bar when their ranks differ by less than the
    Nemenyi critical difference. If the reported family is instead Holm or Finner against
    a control, drawing a Nemenyi bar would let the figure and the table disagree -- the
    figure calling two arms indistinguishable while the table says they differ. So the
    groups are derived from whatever decisions the given result actually made.

    Returns:
        Maximal groups of variants, in rank order, with no significant difference inside.
    """
    ordered = list(mean_ranks.sort_values().index)
    if result.family == "all_pairs":
        cd = float(result.table["critical_difference"].iloc[0])

        groups = []
        for i, first in enumerate(ordered):
            group = [first]
            for other in ordered[i + 1:]:
                if all(abs(float(mean_ranks[m]) - float(mean_ranks[other])) <= cd
                       for m in group):
                    group.append(other)
                else:
                    break
            if not any(set(group) <= set(existing) for existing in groups):
                groups.append(group)
        return [g for g in groups if len(g) > 1]

    # Control family (Holm/Finner against a baseline). Only comparisons TO the baseline
    # were tested, so the only honest grouping is the baseline together with the arms it
    # could not be distinguished from. Arms were never compared to each other, so they
    # must NOT be joined -- a bar spanning two untested arms would assert a result the
    # test never produced. There is therefore exactly one group: the control and its
    # statistical ties.
    base = result.baseline
    significant = {row["variant"] for _, row in result.table.iterrows() if row["reject"]}
    tied_with_control = [base] + [v for v in ordered
                                  if v != base and v not in significant]
    group = [v for v in ordered if v in tied_with_control]
    return [group] if len(group) > 1 else []
