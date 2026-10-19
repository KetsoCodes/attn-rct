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


def _spread(values, min_gap):
    """Push sorted label positions apart so neighbours are at least `min_gap` apart.

    Order is preserved and the group stays centred on the original positions, so a label
    moves as little as it can while no longer overprinting its neighbour.

    Returns:
        Adjusted positions, in the same order as `values`.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    placed = values[order].copy()
    for _ in range(50):
        moved = False
        for i in range(1, len(placed)):
            gap = placed[i] - placed[i - 1]
            if gap < min_gap:
                push = (min_gap - gap) / 2
                placed[i - 1] -= push
                placed[i] += push
                moved = True
        if not moved:
            break
    out = np.empty_like(placed)
    out[order] = placed
    return out


def learning_curve_figure(history: pd.DataFrame, path, chance: dict | None = None,
                          reference: dict | None = None, title: str | None = None):
    """Validation accuracy over training, per arm, with a 95% bootstrap band.

    Seeds are averaged within each (design, arm) first, then the band is a bootstrap over
    designs -- the same unit as every other figure here. The baseline is drawn last and in
    neutral ink so the treatments are read against it.

    Args:
        history: long table with task, design_id, variant, seed, epoch, val_accuracy.
        chance: {task: chance accuracy}, drawn dashed.
        reference: {task: (accuracy, label)}, a published figure drawn as a dotted line.
            For a task that did not train, it is what keeps the y-axis honest: without it
            the axis zooms to the noise around chance and a flat line looks like movement.
    """
    tasks = _tasks(history)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), 3.3),
                             squeeze=False)
    handles = {}
    for ax, task in zip(axes[0], tasks):
        sub = history[history["task"] == task]
        per_design = (sub.groupby(["variant", "design_id", "epoch"])["val_accuracy"]
                      .mean().reset_index())
        arms = [a for a in ARM_ORDER if a in set(per_design["variant"])]
        ends = {}
        for arm in [a for a in arms if a != "flash"] + (["flash"] if "flash" in arms else []):
            curves = per_design[per_design["variant"] == arm].pivot(
                index="epoch", columns="design_id", values="val_accuracy")
            epochs = curves.index.to_numpy() + 1
            mean = curves.mean(axis=1).to_numpy()
            band = np.array([_bootstrap_mean_ci(r.dropna(), n=2000)
                             for _, r in curves.iterrows()])
            ax.fill_between(epochs, band[:, 0], band[:, 1], color=ARM_COLOR[arm],
                            alpha=0.15, linewidth=0)
            (line,) = ax.plot(epochs, mean, color=ARM_COLOR[arm], linewidth=2.0)
            handles[arm] = line
            ends[arm] = (epochs[-1], mean[-1])

        if chance and task in chance:
            ax.axhline(chance[task], color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)))
        if reference and task in reference:
            value, label = reference[task]
            ax.axhline(value, color=INK_2, linewidth=0.9, linestyle=(0, (1, 2)))
            ax.text(1.2, value, f" {label}", fontsize=7, color=INK_2, va="bottom")

        # Keep the y-axis at least 0.10 tall. A task stuck at chance otherwise zooms to a
        # band a few thousandths wide, and sampling noise reads as a learning curve.
        lo, hi = ax.get_ylim()
        if reference and task in reference:
            hi = max(hi, reference[task][0] + 0.03)
        if hi - lo < 0.10:
            mid = (hi + lo) / 2
            lo, hi = mid - 0.05, mid + 0.05
        ax.set_ylim(lo, hi)

        # Direct labels at the line ends, spread so close finishers do not overprint. When
        # every arm finishes in the same place (a task that did not train), five spread
        # labels would only point at one line, so a single label names them all.
        names = list(ends)
        finals = [ends[a][1] for a in names]
        if max(finals) - min(finals) < 0.02 * (hi - lo):
            x_end = max(ends[a][0] for a in names)
            ax.text(x_end + 0.6, float(np.mean(finals)), "All five\narms",
                    fontsize=7, color=INK, va="center")
            names = []
        ys = _spread([ends[a][1] for a in names], min_gap=0.055 * (hi - lo))
        for arm, y in zip(names, ys):
            x_end, y_end = ends[arm]
            ax.annotate(ARM_LABEL[arm].split(" ")[0], (x_end, y_end),
                        xytext=(x_end + 0.6, y), textcoords="data", fontsize=7,
                        color=INK, va="center",
                        arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.6,
                                        shrinkA=0, shrinkB=0)
                        if abs(y - y_end) > 0.01 * (hi - lo) else None)

        ax.set_title(TASK_LABEL.get(task, task), fontsize=10, color=INK, pad=6)
        ax.set_xlabel("Epoch", fontsize=8.5, color=INK)
        ax.set_ylabel("Validation accuracy", fontsize=8.5, color=INK)
        last_epoch = max(e for e, _ in ends.values())
        ax.set_xlim(1, last_epoch + 3.2)
        ax.set_xticks([t for t in (1, 5, 10, 15, 20, 25, 30) if t <= last_epoch])
        _style(ax)

    # One legend for the figure, below the panels, so it never sits on a curve.
    order = [a for a in ARM_ORDER if a in handles]
    fig.legend([handles[a] for a in order], [ARM_LABEL[a] for a in order],
               loc="lower center", ncol=len(order), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))
    if title:
        fig.suptitle(title, fontsize=10.5, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return Path(path)


def _plain_log_ticks(ax):
    """Label a log axis with ordinary numbers (100, 200, 400) instead of 2x10^2."""
    from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
    lo, hi = ax.get_xlim()
    candidates = [20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800,
                  1000, 1500, 2000, 3000, 5000]
    ticks = [t for t in candidates if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def tradeoff_figure(cells: pd.DataFrame, path, title: str | None = None):
    """Accuracy against training cost: the trade-off efficient attention claims to make.

    Each arm is one point per task: mean over designs on both axes, with whiskers spanning
    the interquartile range over designs. Upper-left is better on both counts. All five
    arms share the plot with no positional order, and the categorical hues do not clear the
    all-pairs separation floor, so identity is carried by a direct label on every point and
    every mark is the same neutral colour. Labels are spread apart where points crowd, with
    a thin leader line back to the point they name.
    """
    tasks = [t for t in _tasks(cells)
             if cells.loc[cells["task"] == t, "best_val_accuracy"].notna().any()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), 3.4),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = cells[cells["task"] == task]
        points = {}
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
            points[arm] = (xm, ym)
        ax.set_xscale("log")
        _style(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7, which="major")
        _plain_log_ticks(ax)

        # Place labels in axes-fraction space: cluster points that sit near each other
        # horizontally, then spread each cluster's labels vertically.
        to_axes = ax.transData + ax.transAxes.inverted()
        frac = {arm: to_axes.transform(xy) for arm, xy in points.items()}
        arms = sorted(frac, key=lambda a: frac[a][0])
        clusters, current = [], [arms[0]]
        for arm in arms[1:]:
            if frac[arm][0] - frac[current[-1]][0] < 0.14:
                current.append(arm)
            else:
                clusters.append(current)
                current = [arm]
        clusters.append(current)
        for cluster in clusters:
            ys = _spread([frac[a][1] + 0.05 for a in cluster], min_gap=0.095)
            for arm, ly in zip(cluster, ys):
                fx, fy = frac[arm]
                lx = fx + 0.045
                name = ARM_LABEL[arm].replace(" (baseline)", "*")
                moved = abs(ly - (fy + 0.05)) > 0.005
                ax.annotate(name, (fx, fy), xycoords="axes fraction",
                            xytext=(lx, ly), textcoords="axes fraction",
                            fontsize=8, color=INK, va="center",
                            arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.6,
                                            shrinkA=0, shrinkB=3) if moved else None)

        ax.set_title(TASK_LABEL.get(task, task), fontsize=10, color=INK, pad=6)
        ax.set_xlabel("Training time per epoch (s, log scale)", fontsize=8.5, color=INK)
        ax.set_ylabel("Best validation accuracy", fontsize=8.5, color=INK)
    fig.text(0.01, -0.02, "* baseline.  Point = mean over designs; whiskers = IQR over "
             "designs.  Up and to the left is better.", fontsize=7, color=INK_2)
    if title:
        fig.suptitle(title, fontsize=10.5, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return Path(path)


def significant_vs_baseline(cells: pd.DataFrame, metric: str, baseline="flash",
                            method="finner", alpha=0.05) -> dict:
    """Which arms the pre-specified post-hoc declares different from the baseline.

    Computed per scope (pooled, then each task) with the same ranking and correction as
    run.py, so the stars on a figure can never disagree with the results table.

    Returns:
        {scope: set of arms}.
    """
    from . import posthoc
    scopes = {"pooled": cells}
    if cells["task"].nunique() > 1:
        scopes.update({t: cells[cells["task"] == t] for t in _tasks(cells)})
    out = {}
    for scope, sub in scopes.items():
        ranked = aggregate.rank_within_blocks(sub, metric)
        n_blocks = sub[aggregate.BLOCK_KEYS].drop_duplicates().shape[0]
        result = posthoc.compare_to_control(aggregate.mean_ranks(ranked), baseline,
                                            n_blocks, metric, method=method, alpha=alpha)
        out[scope] = set(result.table.loc[result.table["reject"], "variant"])
    return out


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
                             "Difference from Flash (accuracy points)", scale=100.0,
                             significant=significant_vs_baseline(acc_cells,
                                                                 "best_val_accuracy"))
    for metric, label in (("mean_train_seconds_per_epoch", "Seconds per epoch"),
                          ("peak_memory_mb", "Peak GPU memory (MB)")):
        if metric in cells.columns:
            per_design_figure(cells, metric, out / f"fig_{metric}_per_design.{ext}", label)
    if "mean_train_seconds_per_epoch" in cells.columns:
        tradeoff_figure(acc_cells, out / f"fig_tradeoff.{ext}")

    history = load_histories(args.results_dir, args.phase)
    if not history.empty:
        # LRA's published Transformer result for Pathfinder, so a reader sees how far the
        # flat line sits from what the task can reach (Tay et al., 2021, Table 1).
        reference = {"pathfinder": (0.714, "LRA Transformer, 71.4%")}
        learning_curve_figure(history, out / f"fig_learning_curves.{ext}", chance=chance,
                              reference=reference)

    for metric in metrics:
        source = acc_cells if metric == "best_val_accuracy" else cells
        summary_table(source, metric).round(4).to_csv(out / f"table_{metric}.csv")

    print(f"wrote figures to {out}/")


if __name__ == "__main__":
    main()
