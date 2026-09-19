"""Visualizations and summary tables for the experimental results.

Generates Critical-Difference (CD) diagrams, results tables, and cross-task 
ranking plots. Visualizations are built directly from prior statistical 
decisions (e.g., Finner post-hoc cliques) to ensure the figures perfectly 
match the numerical reports without recomputing any metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # Headless: render to files, never to a display
import matplotlib.pyplot as plt
import pandas as pd

from . import posthoc


def _draw_bar(ax, mean_ranks, run, y):
    """Draws one connecting bar across a run of rank-adjacent, indistinguishable arms."""
    if len(run) < 2:
        return
    positions = [mean_ranks[v] for v in run]
    ax.plot([min(positions) - 0.05, max(positions) + 0.05], [y, y],
            "k-", linewidth=3.0, solid_capstyle="round")


def critical_difference_diagram(mean_ranks: pd.Series, posthoc_result,
                                title: str, path, cd: float | None = None,
                                figsize=(8.0, 2.6)):
    """Draws a critical-difference diagram from mean ranks and post-hoc results.

    Args:
        mean_ranks: Mean rank per variant.
        posthoc_result: PostHocResult defining the indistinguishable groups.
        title: Figure title.
        path: Output file path (.svg or .png).
        cd: Nemenyi critical difference for the scale bar (computed if omitted).
        figsize: Figure dimensions in inches.
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
    ax.set_xlim(low - 0.5, high + 0.5)
    ax.set_ylim(-0.15, 1)
    ax.axis("off")

    axis_y = 0.75
    ax.plot([low, high], [axis_y, axis_y], "k-", linewidth=1.0)
    for tick in range(low, high + 1):
        ax.plot([tick, tick], [axis_y, axis_y + 0.03], "k-", linewidth=1.0)
        ax.text(tick, axis_y + 0.07, str(tick), ha="center", va="bottom", fontsize=9)

    # Stagger labels to prevent text overlap for closely ranked variants.
    depths = []
    last_rank = None
    level = 0
    for rank in ranks:
        if last_rank is not None and abs(rank - last_rank) < 0.4:
            level += 1
        else:
            level = 0
        depths.append(level)
        last_rank = rank
        
    label_ys = [0.45, 0.28, 0.11, -0.06]
    for i, (variant, rank) in enumerate(zip(variants, ranks)):
        y = label_ys[min(depths[i], len(label_ys) - 1)]
        ax.plot([rank, rank], [axis_y, y + 0.04], "k-", linewidth=0.8)
        ax.text(rank, y, f"{variant}\n{rank:.2f}", ha="center", va="top", fontsize=9)

    # Draw connecting bars for indistinguishable groups based on post-hoc cliques.
    # We bridge only runs of rank-adjacent members to avoid visually spanning excluded arms.
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

    # Nemenyi CD scale bar for reference (all-pairs threshold).
    ax.plot([low, low + cd], [0.93, 0.93], "k-", linewidth=1.5)
    ax.plot([low, low], [0.91, 0.95], "k-", linewidth=1.0)
    ax.plot([low + cd, low + cd], [0.91, 0.95], "k-", linewidth=1.0)
    ax.text(low + cd / 2, 0.97, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title, fontsize=11, pad=2)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return Path(path)


def results_table(omnibus_result, posthoc_result, effect_result) -> pd.DataFrame:
    """Assembles a tidy DataFrame joining ranks, adjusted p-values, and effect sizes.

    Returns:
        A DataFrame ordered by mean rank, with the baseline row explicitly marked.
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
    """Plots mean ranks per variant across tasks to highlight performance shifts.

    Args:
        per_task_ranks: Dictionary mapping task names to mean_ranks Series.
        metric: Metric name for the y-axis label.
        title: Figure title.
        path: Output file path.
        figsize: Figure dimensions in inches.
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
    ax.invert_yaxis()  # Rank 1 at the top (visually higher is better)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.margins(x=0.15)
    
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return Path(path)