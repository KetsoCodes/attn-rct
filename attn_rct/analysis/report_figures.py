"""Generates supervisor-facing figures in the style of Pretorius et al. (2020).

These plots show individual-level distributions and paired-effect intervals to give a clear 
view of the data alongside the statistical tests. 

Key principles used here:
  - Seeds are averaged first (they are just noise control, not independent observations).
  - Paired lines connect designs across arms to highlight within-design differences.
  - Effects are shown in raw units (e.g., accuracy points or seconds) for readability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.subplots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import aggregate, collect

# Baseline first, then treatments. Kept fixed so colours and positions stay consistent.
ARM_ORDER = ["flash", "vanilla", "linformer", "linear", "sparse"]
ARM_COLOR = {
    "flash": "#52514e",
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
TASK_LABEL = {
    "listops": "ListOps", 
    "cifar": "CIFAR-10 (Image)",
    "pathfinder": "Pathfinder", 
    "text": "Text (IMDb)"
}

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def _style(ax):
    """Minimizes axis distractions so the data stands out."""
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
    """Calculates the mean metric per arm and task, plus the percentage change from the baseline."""
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
    """Percentile bootstrap CI for a mean."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return (float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)))


def paired_differences(cells: pd.DataFrame, metric: str, baseline="flash",
                       scope_task=None) -> pd.DataFrame:
    """Calculates the paired difference from the baseline for each arm, with a bootstrap CI."""
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
    """Plots every design's performance across arms, connecting paired designs with thin lines."""
    tasks = _tasks(cells)
    arms = _arms_present(cells)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), 3.6),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = cells[cells["task"] == task]
        matrix = aggregate.value_matrix(sub, metric).reindex(columns=arms)
        xs = np.arange(len(arms))

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
    """Draws a forest plot of each arm's mean paired difference from the baseline (95% CI)."""
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
            ax.text(r.ci_high * scale, y,
                    f"  {r.mean_diff * scale:+.1f}" + (" *" if marked else ""),
                    fontsize=7.5, color=INK, va="center", ha="left")
            
        ax.axvline(0, color=INK_2, linewidth=0.9)
        ax.set_ylim(-0.6, len(arms) - 0.4)
        
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.22 * (hi - lo)) 
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
    """Plots validation accuracy over epochs with a 95% bootstrap band over designs."""
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
    """Plots best validation accuracy against training cost to show the efficiency trade-off."""
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
# CLI
# --------------------------------------------------------------------------------------

def load_histories(results_dir, phase="full") -> pd.DataFrame:
    """Extracts per-epoch validation accuracy for full-phase results."""
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