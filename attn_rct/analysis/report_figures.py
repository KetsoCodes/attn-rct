"""Supervisor-facing figures, in the presentation style of Pretorius et al. (2020).

The Demsar pipeline (run.py) produces the figures the analysis plan pre-specifies: a
critical-difference diagram and a results table per scope. Those answer the inferential
question. They do not show a reader what the data looks like, which is what a supervisor
needs first, and what Pretorius et al. show with individual-level distributions and
paired-effect intervals. This module draws those views.

Every figure here is a VIEW of the same seed-averaged cells the inference uses, so nothing
can contradict the statistics:

  * The unit of observation is the (task, design) block, never the seed. Seeds are noise
    control and are averaged first, exactly as aggregate.aggregate_seeds does, because
    plotting them as separate points would show 3x the evidence the test actually has.
  * Paired structure is drawn, not hidden. Each design is joined across arms by a thin
    line, because the comparison is within-design and a cloud of unpaired dots would
    discard the very thing that gives the design its power.
  * Effects are shown in raw units (accuracy points, seconds) alongside the standardised
    effect. A Cohen's d_z of 4 means little to a reader; "13.8 points better on every
    design" is the claim in the units the reader already uses.

Colour: the four treatment arms take a validated categorical order (blue, orange, violet,
aqua; passes adjacent CVD separation with worst Delta E 24.7) and the baseline is a neutral
grey, so the control reads as the reference rather than as a fifth competitor. In the one
figure where all five arms share a plot without positional order (the trade-off scatter),
these hues do NOT clear the all-pairs floor (blue vs violet, Delta E 4.5 deutan), so that
figure uses a single neutral mark and labels every point directly instead.

Usage:
    python -m attn_rct.analysis.report_figures --results-dir results --out report/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import aggregate, collect

# Baseline first, then the treatments in the manifest's order. Fixed everywhere, so an arm
# sits in the same place and wears the same colour in every figure.
ARM_ORDER = ["flash", "vanilla", "linformer", "linear", "sparse"]
ARM_COLOR = {
    "flash": "#52514e",       # baseline: neutral, so it reads as the reference
    "vanilla": "#2a78d6",
    "linformer": "#eb6834",
    "linear": "#6250d6",
    "sparse": "#1baf7a",
}
ARM_LABEL = {
    "flash": "Flash (baseline)",
    "vanilla": "Vanilla",
    "linformer": "Linformer",
    "linear": "Linear",
    "sparse": "Sparse",
}
TASK_LABEL = {"listops": "ListOps", "cifar": "CIFAR-10 (Image)",
              "pathfinder": "Pathfinder", "text": "Text (IMDb)"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def _style(ax):
    """Recessive axes: the data is the figure, not the frame around it."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _arms_present(cells):
    return [a for a in ARM_ORDER if a in set(cells["variant"])]


def _tasks(cells):
    order = ["listops", "cifar", "text", "pathfinder"]
    present = set(cells["task"])
    return [t for t in order if t in present] + sorted(present - set(order))


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------

def summary_table(cells: pd.DataFrame, metric: str, baseline="flash") -> pd.DataFrame:
    """Mean of a metric per arm and task, with the change relative to the baseline.

    The direct analogue of a "Table 1": one number per arm per task, so the reader sees the
    size of the effect before any test. The mean is over designs (blocks), which is the
    population the study generalises to.

    Returns:
        DataFrame indexed by arm, with one `<task>` and one `<task>_pct` column per task.
    """
    out = {}
    for task in _tasks(cells):
        sub = cells[cells["task"] == task]
        means = sub.groupby("variant")[metric].mean()
        out[task] = means
        if baseline in means:
            out[f"{task}_pct"] = 100.0 * (means - means[baseline]) / means[baseline]
    table = pd.DataFrame(out)
    return table.reindex([a for a in ARM_ORDER if a in table.index])


def _bootstrap_mean_ci(values, n=10_000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a mean, resampling blocks."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return (float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)))


def paired_differences(cells: pd.DataFrame, metric: str, baseline="flash",
                       scope_task=None) -> pd.DataFrame:
    """Per-arm paired difference from the baseline, in raw units, with a bootstrap CI.

    Differences are taken WITHIN each block (same design, same task) and then summarised
    over blocks, which is what makes this a paired comparison rather than a difference of
    two independent means.

    Returns:
        One row per non-baseline arm: mean_diff, ci_low, ci_high, n_blocks, n_better.
    """
    sub = cells if scope_task is None else cells[cells["task"] == scope_task]
    matrix = aggregate.value_matrix(sub, metric)
    rows = []
    for arm in [a for a in ARM_ORDER if a in matrix.columns and a != baseline]:
        diffs = (matrix[arm] - matrix[baseline]).to_numpy()
        low, high = _bootstrap_mean_ci(diffs)
        rows.append({
            "variant": arm, "mean_diff": float(diffs.mean()),
            "ci_low": low, "ci_high": high,
            "n_blocks": len(diffs), "n_better": int((diffs > 0).sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

def per_design_figure(cells: pd.DataFrame, metric: str, path, ylabel: str,
                      chance: dict | None = None, title: str | None = None):
    """Every design's value under every arm, with the pairing drawn.

    Box shows median and IQR over designs; each dot is one design (seed-averaged); thin
    grey lines join the same design across arms. The lines are the point: in a paired
    design the question is whether an arm wins *within* designs, and a box plot alone
    cannot show that. Crossing lines mean the ranking depends on the configuration;
    parallel lines mean it does not.
    """
    tasks = _tasks(cells)
    arms = _arms_present(cells)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), 3.6),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = cells[cells["task"] == task]
        matrix = aggregate.value_matrix(sub, metric).reindex(columns=arms)
        xs = np.arange(len(arms))

        # Pairing lines first, so the marks sit on top of them.
        for _, row in matrix.iterrows():
            ax.plot(xs, row.values, color="#bdbcb6", linewidth=0.7, zorder=1)

        for x, arm in zip(xs, arms):
            vals = matrix[arm].dropna().to_numpy()
            ax.boxplot(vals, positions=[x], widths=0.5, showfliers=False,
                       patch_artist=True, zorder=2,
                       boxprops=dict(facecolor="none", edgecolor=INK_2, linewidth=0.9),
                       medianprops=dict(color=INK, linewidth=1.4),
                       whiskerprops=dict(color=INK_2, linewidth=0.9),
                       capprops=dict(color=INK_2, linewidth=0.9))
            ax.scatter(np.full(len(vals), x), vals, s=26, color=ARM_COLOR[arm],
                       edgecolor="white", linewidth=0.8, zorder=3)

        if chance and task in chance:
            ax.axhline(chance[task], color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)),
                       zorder=0)
            ax.text(len(arms) - 0.45, chance[task], " chance", va="bottom", ha="right",
                    fontsize=7, color=INK_2)

        ax.set_xticks(xs)
        ax.set_xticklabels([ARM_LABEL[a].replace(" (baseline)", "\n(baseline)")
                            for a in arms], fontsize=8, color=INK)
        ax.set_title(TASK_LABEL.get(task, task), fontsize=10, color=INK, pad=6)
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK)
        _style(ax)

    if title:
        fig.suptitle(title, fontsize=10.5, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return Path(path)


def paired_difference_figure(cells: pd.DataFrame, metric: str, path, xlabel: str,
                             baseline="flash", scale=1.0, significant=None,
                             title: str | None = None):
    """Forest plot: each arm's mean paired difference from the baseline, with a 95% CI.

    One panel per scope (pooled, then each task). The interval is a percentile bootstrap
    over blocks, so it is a description of the effect's size and precision. It is NOT the
    decision of record -- that is the pre-specified Finner-corrected rank test -- and the
    two can disagree near the boundary, which is why significance is marked separately from
    the interval rather than read off it.

    Args:
        scale: multiply differences by this for display (100 turns accuracy into points).
        significant: {scope: set(arms)} that the pre-specified test declared significant.
    """
    scopes = ["pooled"] + _tasks(cells) if cells["task"].nunique() > 1 else _tasks(cells)
    fig, axes = plt.subplots(1, len(scopes), figsize=(3.0 * len(scopes), 2.6),
                             squeeze=False, sharey=True)
    arms = [a for a in _arms_present(cells) if a != baseline]
    for ax, scope in zip(axes[0], scopes):
        task = None if scope == "pooled" else scope
        diffs = paired_differences(cells, metric, baseline, task).set_index("variant")
        ys = np.arange(len(arms))[::-1]
        for y, arm in zip(ys, arms):
            r = diffs.loc[arm]
            ax.plot([r.ci_low * scale, r.ci_high * scale], [y, y],
                    color=ARM_COLOR[arm], linewidth=2.0, solid_capstyle="round")
            ax.scatter([r.mean_diff * scale], [y], s=46, color=ARM_COLOR[arm],
                       edgecolor="white", linewidth=1.0, zorder=3)
            marked = significant and arm in significant.get(scope, set())
            # Beside the interval's right end, on the same row, so the label can never
            # climb into the panel title.
            ax.text(r.ci_high * scale, y,
                    f"  {r.mean_diff * scale:+.1f}" + (" *" if marked else ""),
                    fontsize=7.5, color=INK, va="center", ha="left")
        ax.axvline(0, color=INK_2, linewidth=0.9)
        ax.set_ylim(-0.6, len(arms) - 0.4)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.22 * (hi - lo))      # room for the right-hand labels
        ax.set_yticks(ys)
        ax.set_yticklabels([ARM_LABEL[a] for a in arms], fontsize=8, color=INK)
        name = "Pooled" if scope == "pooled" else TASK_LABEL.get(scope, scope)
        n = int(diffs["n_blocks"].iloc[0])
        ax.set_title(f"{name}  (n = {n} designs)", fontsize=9.5, color=INK, pad=6)
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK)
        _style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)

    if title:
        fig.suptitle(title, fontsize=10.5, color=INK, y=1.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return Path(path)


def learning_curve_figure(history: pd.DataFrame, path, chance: dict | None = None,
                          title: str | None = None):
    """Validation accuracy over training, per arm, with a 95% bootstrap band.

    Seeds are averaged within each (design, arm) first, then the band is a bootstrap over
    designs -- the same unit as every other figure here. The baseline is drawn last and in
    neutral ink so the treatments are read against it.

    Args:
        history: long table with task, design_id, variant, seed, epoch, val_accuracy.
    """
    tasks = _tasks(history)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), 3.3),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = history[history["task"] == task]
        per_design = (sub.groupby(["variant", "design_id", "epoch"])["val_accuracy"]
                      .mean().reset_index())
        arms = [a for a in ARM_ORDER if a in set(per_design["variant"])]
        for arm in [a for a in arms if a != "flash"] + (["flash"] if "flash" in arms else []):
            curves = per_design[per_design["variant"] == arm].pivot(
                index="epoch", columns="design_id", values="val_accuracy")
            epochs = curves.index.to_numpy() + 1
            mean = curves.mean(axis=1).to_numpy()
            band = np.array([_bootstrap_mean_ci(r.dropna(), n=2000)
                             for _, r in curves.iterrows()])
            ax.fill_between(epochs, band[:, 0], band[:, 1], color=ARM_COLOR[arm],
                            alpha=0.15, linewidth=0)
            ax.plot(epochs, mean, color=ARM_COLOR[arm], linewidth=2.0,
                    label=ARM_LABEL[arm])
            # Direct label at the line's end, in neutral ink beside a coloured mark.
            ax.text(epochs[-1] + 0.3, mean[-1], ARM_LABEL[arm].split(" ")[0],
                    fontsize=7, color=INK, va="center")
        if chance and task in chance:
            ax.axhline(chance[task], color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_title(TASK_LABEL.get(task, task), fontsize=10, color=INK, pad=6)
        ax.set_xlabel("Epoch", fontsize=8.5, color=INK)
        ax.set_ylabel("Validation accuracy", fontsize=8.5, color=INK)
        ax.set_xlim(left=1)
        _style(ax)
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower right")
    if title:
        fig.suptitle(title, fontsize=10.5, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return Path(path)


def tradeoff_figure(cells: pd.DataFrame, path, title: str | None = None):
    """Accuracy against training cost: the trade-off efficient attention claims to make.

    Each arm is one point per task: mean over designs on both axes, with whiskers spanning
    the interquartile range over designs. Upper-left is better on both counts. All five
    arms share the plot with no positional order, and the categorical hues do not clear the
    all-pairs separation floor, so identity is carried by a direct label on every point and
    every mark is the same neutral colour.
    """
    tasks = [t for t in _tasks(cells)
             if cells.loc[cells["task"] == t, "best_val_accuracy"].notna().any()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), 3.4),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = cells[cells["task"] == task]
        for arm in _arms_present(sub):
            a = sub[sub["variant"] == arm]
            x = a["mean_train_seconds_per_epoch"]
            y = a["best_val_accuracy"]
            xm, ym = x.mean(), y.mean()
            ax.plot([x.quantile(.25), x.quantile(.75)], [ym, ym], color=INK_2,
                    linewidth=1.0)
            ax.plot([xm, xm], [y.quantile(.25), y.quantile(.75)], color=INK_2,
                    linewidth=1.0)
            ax.scatter([xm], [ym], s=48, color=INK, edgecolor="white", linewidth=1.0,
                       zorder=3)
            ax.annotate(ARM_LABEL[arm].replace(" (baseline)", "*"), (xm, ym),
                        xytext=(6, 5), textcoords="offset points", fontsize=8,
                        color=INK)
        ax.set_xscale("log")
        ax.set_title(TASK_LABEL.get(task, task), fontsize=10, color=INK, pad=6)
        ax.set_xlabel("Training time per epoch (s, log scale)", fontsize=8.5, color=INK)
        ax.set_ylabel("Best validation accuracy", fontsize=8.5, color=INK)
        _style(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7, which="both")
    fig.text(0.01, -0.02, "* baseline.  Point = mean over designs; whiskers = IQR over "
             "designs.  Up and to the left is better.", fontsize=7, color=INK_2)
    if title:
        fig.suptitle(title, fontsize=10.5, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return Path(path)


# --------------------------------------------------------------------------------------
# CLI: everything from results/
# --------------------------------------------------------------------------------------

def load_histories(results_dir, phase="full") -> pd.DataFrame:
    """Per-epoch validation accuracy for every full-phase result, as a long table."""
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("phase") != phase or "history" not in payload:
            continue
        task = payload.get("task") or payload.get("config", {}).get("task", "listops")
        for rec in payload["history"]:
            rows.append({"task": task, "design_id": payload["design_id"],
                         "variant": payload["variant"], "seed": payload["seed"],
                         "epoch": rec["epoch"], "val_accuracy": rec["val_accuracy"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="report/figures")
    parser.add_argument("--phase", default="full")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ext = args.format

    df, report = collect.load_results(args.results_dir, phase=args.phase)
    print(report.summary())
    keep = collect.require_complete(df, report)
    metrics = [m for m in ("best_val_accuracy", "mean_train_seconds_per_epoch",
                           "peak_memory_mb") if m in keep.columns]
    cells = aggregate.aggregate_seeds(keep, metrics)

    acc_cells, dropped = aggregate.drop_degenerate_blocks(cells, "best_val_accuracy")
    if dropped:
        print(f"accuracy figures exclude {len(dropped)} degenerate block(s): "
              + ", ".join(f"{t}/d{d:03d}" for t, d in dropped))

    chance = aggregate.TASK_CHANCE
    per_design_figure(acc_cells, "best_val_accuracy", out / f"fig_accuracy_per_design.{ext}",
                      "Best validation accuracy", chance=chance)
    paired_difference_figure(acc_cells, "best_val_accuracy",
                             out / f"fig_accuracy_paired_diff.{ext}",
                             "Difference from Flash (accuracy points)", scale=100.0)
    for metric, label in (("mean_train_seconds_per_epoch", "Seconds per epoch"),
                          ("peak_memory_mb", "Peak GPU memory (MB)")):
        if metric in cells.columns:
            per_design_figure(cells, metric, out / f"fig_{metric}_per_design.{ext}", label)
    if "mean_train_seconds_per_epoch" in cells.columns:
        tradeoff_figure(acc_cells, out / f"fig_tradeoff.{ext}")

    history = load_histories(args.results_dir, args.phase)
    if not history.empty:
        learning_curve_figure(history, out / f"fig_learning_curves.{ext}", chance=chance)

    for metric in metrics:
        source = acc_cells if metric == "best_val_accuracy" else cells
        summary_table(source, metric).round(4).to_csv(out / f"table_{metric}.csv")

    print(f"wrote figures to {out}/")


if __name__ == "__main__":
    main()
