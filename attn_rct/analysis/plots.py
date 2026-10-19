"""Critical-difference diagrams and the summary results table.

The critical-difference diagram is the standard way to show a Demsar comparison: variants
sit on a rank axis, best on the left, and a bar joins any set the post-hoc cannot tell
apart. The one subtlety that matters here is which post-hoc the bar reflects.

The textbook CD bar is the Nemenyi critical difference, an all-pairs quantity. This study
reports Finner comparisons against a control, which is a different and more powerful test.
If the diagram drew a Nemenyi bar over Finner results, the figure could join two arms the
table separates -- the picture contradicting the numbers. So the connecting groups come
from posthoc.cliques, which derives them from the decisions the reported post-hoc actually
made. The Nemenyi critical difference is still drawn, as a labelled scale bar, so the
reader can see the all-pairs threshold; it just is not what the connecting lines encode.

Nothing here recomputes a statistic. The figures and the table are views of results the
omnibus, post-hoc and effect stages already produced, so a number can never appear in a
plot that disagrees with the table it came from.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: render to files, never to a display
import matplotlib.pyplot as plt
import pandas as pd

from . import posthoc


def _draw_bar(ax, mean_ranks, run, y):
    """Draw one connecting bar across a run of rank-adjacent, indistinguishable arms."""
    if len(run) < 2:
        return
    positions = [mean_ranks[v] for v in run]
    ax.plot([min(positions) - 0.05, max(positions) + 0.05], [y, y],
            "k-", linewidth=3.0, solid_capstyle="round")


def critical_difference_diagram(mean_ranks: pd.Series, posthoc_result,
                                title: str, path, cd: float | None = None,
                                figsize=(8.0, 2.6)):
    """Draw a critical-difference diagram from a mean-rank series and a post-hoc result.

    Args:
        mean_ranks: mean rank per variant (aggregate.mean_ranks).
        posthoc_result: the PostHocResult whose decisions define the connecting groups.
        title: figure title.
        path: output file (.svg or .png).
        cd: Nemenyi critical difference to draw as a scale bar; computed if omitted.
        figsize: figure size in inches.

    Returns:
        The path written.
    """
    ordered = mean_ranks.sort_values()
    variants = list(ordered.index)
    ranks = list(ordered.values)
    k = len(variants)
    low, high = 1, k

    if cd is None:
        cd = posthoc.critical_difference(k, posthoc_result.n_blocks,
                                         alpha=posthoc_result.alpha)

    groups = posthoc.cliques(mean_ranks, posthoc_result)

    fig, ax = plt.subplots(figsize=figsize)
    margin = 1.6                      # room either side for the elbowed labels
    ax.set_xlim(low - margin, high + margin)
    ax.set_ylim(-0.05, 1.12)
    ax.axis("off")

    axis_y = 0.75
    ax.plot([low, high], [axis_y, axis_y], "k-", linewidth=1.0)
    for tick in range(low, high + 1):
        ax.plot([tick, tick], [axis_y, axis_y + 0.03], "k-", linewidth=1.0)
        ax.text(tick, axis_y + 0.07, str(tick), ha="center", va="bottom", fontsize=9)

    # Demsar's layout: the better half of the arms elbow down and out to labels on the
    # LEFT, the worse half to labels on the RIGHT, each successive arm one row lower. A
    # label therefore never sits on a stem, however close two mean ranks are -- an earlier
    # version centred each label on its own stem and tied ranks overprinted the lines.
    n_left = (k + 1) // 2
    row_step = 0.62 / max(n_left, k - n_left, 1)
    for i, (variant, rank) in enumerate(zip(variants, ranks)):
        left = i < n_left
        row = i if left else (k - 1 - i)          # outermost arms on the top row
        y = axis_y - 0.22 - row * row_step
        x_end = low - 0.25 if left else high + 0.25
        ax.plot([rank, rank], [axis_y, y], "k-", linewidth=0.8)
        ax.plot([rank, x_end], [y, y], "k-", linewidth=0.8)
        ax.text(x_end + (-0.06 if left else 0.06), y, f"{variant} ({rank:.2f})",
                ha="right" if left else "left", va="center", fontsize=9)

    # Connecting bars for groups the post-hoc cannot separate. A group may not be
    # contiguous in rank order (a control family can tie the baseline with a distant arm
    # while separating a nearer one), and a bar drawn straight across would visually span
    # the arm in between. So within each group we bridge only runs of rank-adjacent
    # members, which never crosses an excluded arm.
    rank_order = list(mean_ranks.sort_values().index)
    bar_y = axis_y - 0.06
    depth = 0
    for group in groups:
        members = [v for v in rank_order if v in group]
        run = [members[0]]
        for v in members[1:]:
            if rank_order.index(v) == rank_order.index(run[-1]) + 1:
                run.append(v)
            else:
                _draw_bar(ax, mean_ranks, run, bar_y - depth * 0.04)
                depth += 1
                run = [v]
        _draw_bar(ax, mean_ranks, run, bar_y - depth * 0.04)
        depth += 1

    # CD scale bar, top-left, labelled -- the all-pairs threshold for reference.
    ax.plot([low, low + cd], [0.98, 0.98], "k-", linewidth=1.5)
    ax.plot([low, low], [0.96, 1.00], "k-", linewidth=1.0)
    ax.plot([low + cd, low + cd], [0.96, 1.00], "k-", linewidth=1.0)
    ax.text(low + cd + 0.08, 0.98, f"Nemenyi CD = {cd:.2f}", ha="left", va="center",
            fontsize=8)

    ax.set_title(title, fontsize=11, pad=8)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return Path(path)


def results_table(omnibus_result, posthoc_result, effect_result) -> pd.DataFrame:
    """Assemble one tidy table joining ranks, adjusted p-values, and effect sizes.

    Every column comes from a stage that already computed it, so the table cannot drift
    from the figures. One row per variant; the baseline row is marked.

    Returns:
        A DataFrame ordered by mean rank.
    """
    baseline = posthoc_result.baseline
    ranks = omnibus_result.mean_ranks

    ph = posthoc_result.table.set_index("variant")
    ef = effect_result.table.set_index("variant")

    rows = []
    for variant in sorted(ranks, key=lambda v: ranks[v]):
        row = {
            "variant": variant,
            "mean_rank": round(ranks[variant], 3),
            "is_baseline": variant == baseline,
        }
        if variant in ph.index:
            row["p_adjusted"] = round(float(ph.loc[variant, "p_adjusted"]), 4)
            row["significant_vs_baseline"] = bool(ph.loc[variant, "reject"])
        if variant in ef.index:
            row["cohens_dz"] = round(float(ef.loc[variant, "cohens_dz"]), 3)
            row["ci_low"] = round(float(ef.loc[variant, "ci_low"]), 3)
            row["ci_high"] = round(float(ef.loc[variant, "ci_high"]), 3)
            if "geometric_ratio" in ef.columns:
                row["geometric_ratio"] = round(float(ef.loc[variant, "geometric_ratio"]), 3)
            row["favours"] = ef.loc[variant, "favours"]
        rows.append(row)

    return pd.DataFrame(rows)


def task_comparison_figure(per_task_ranks: dict, metric: str, title: str, path,
                           figsize=(8.0, 4.0)):
    """Plot mean ranks per variant across tasks side by side, to show cross-task change.

    Args:
        per_task_ranks: {task_name: mean_ranks Series} for one metric.
        metric: metric name, for the axis label.
        title: figure title.
        path: output file.

    Returns:
        The path written.

    The point of this figure is the one thing pooling hides: whether the ranking is the
    same on every task or moves. A variant whose line stays flat ranks consistently; a
    variant whose line crosses others is task-dependent, which is the finding the study
    most wants to surface.
    """
    tasks = list(per_task_ranks)
    variants = sorted(next(iter(per_task_ranks.values())).index)

    fig, ax = plt.subplots(figsize=figsize)
    for variant in variants:
        ys = [per_task_ranks[t][variant] for t in tasks]
        ax.plot(range(len(tasks)), ys, marker="o", linewidth=1.5, label=variant)
        ax.text(len(tasks) - 1 + 0.05, ys[-1], variant, va="center", fontsize=9)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks)
    ax.set_ylabel(f"mean rank ({metric})\nlower is better")
    ax.invert_yaxis()            # rank 1 at the top, so "up" reads as "better"
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.margins(x=0.15)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return Path(path)
