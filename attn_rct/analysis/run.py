"""Run the whole Demsar pipeline and write the report.

This is the one command that turns a directory of result files into the artifacts a
write-up needs: for every metric, at every scope (pooled and each task), it collects,
checks, aggregates, runs the omnibus, the post-hoc against the baseline, and the effect
sizes, then writes a critical-difference diagram, a results table, and a machine-readable
summary. Per-metric cross-task figures show whether the ranking moves between tasks.

Nothing is recomputed anywhere but in the analysis stages, so every number in every
figure traces back to the same table. The run refuses rather than guesses: an incomplete
matrix or a failed integrity check stops it, because a partial or broken matrix silently
analysed is the failure mode the whole design exists to prevent.

    python -m attn_rct.analysis.run --results-dir results --out report
    python -m attn_rct.analysis.run --results-dir results --baseline flash --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import aggregate, collect, effects, omnibus, plots, posthoc

METRICS = ["best_val_accuracy", "mean_train_seconds_per_epoch", "peak_memory_mb"]


def analyse_scope(cells, metric, baseline, alpha, method, n_permutations, n_bootstrap):
    """Run omnibus, post-hoc and effects for one metric on one set of cells.

    Returns:
        Dict of the three results, or None if the omnibus does not reject (in which case
        post-hoc is deliberately not run, per the plan).
    """
    direction = aggregate.METRIC_DIRECTION[metric]
    ranked = aggregate.rank_within_blocks(cells, metric)
    rank_matrix = aggregate.rank_matrix(ranked)
    mean_ranks = aggregate.mean_ranks(ranked)

    om = omnibus.run_omnibus(rank_matrix, metric, alpha=alpha,
                             n_permutations=n_permutations)

    result = {"omnibus": om, "mean_ranks": mean_ranks, "posthoc": None, "effects": None}
    # Effect sizes are reported regardless of the omnibus, because a non-rejection at
    # this N is not evidence of equivalence. Post-hoc is only run on a rejection.
    result["effects"] = effects.compute_effects(
        aggregate.value_matrix(cells, metric), baseline, metric, direction,
        n_bootstrap=n_bootstrap, alpha=alpha,
    )
    if om.reject:
        result["posthoc"] = posthoc.compare_to_control(
            mean_ranks, baseline, len(rank_matrix), metric, method=method, alpha=alpha,
        )
    return result


def _scope_summary(metric, res) -> dict:
    """A JSON-safe summary of one metric's analysis at one scope."""
    om = res["omnibus"]
    summary = {
        "metric": metric,
        "n_blocks": om.n_blocks,
        "mean_ranks": {k: round(v, 3) for k, v in om.mean_ranks.items()},
        "friedman_chi2": round(om.friedman_chi2, 4),
        "friedman_p": om.friedman_p,
        "permutation_p": om.permutation_p,
        "reject": om.reject,
    }
    if res["posthoc"] is not None:
        summary["significant_vs_baseline"] = list(res["posthoc"].significant()["variant"])
    if res["effects"] is not None:
        summary["effects"] = {
            r["variant"]: {
                "cohens_dz": round(float(r["cohens_dz"]), 3),
                "ci": [round(float(r["ci_low"]), 3), round(float(r["ci_high"]), 3)],
                **({"geometric_ratio": round(float(r["geometric_ratio"]), 3)}
                   if "geometric_ratio" in r else {}),
            }
            for _, r in res["effects"].table.iterrows()
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the Demsar analysis pipeline.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="report")
    parser.add_argument("--phase", default="full")
    parser.add_argument("--baseline", default="flash")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--method", default="finner", choices=["finner", "holm"])
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--figure-format", default="svg", choices=["svg", "png"])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, report = collect.load_results(args.results_dir, phase=args.phase)
    print(report.summary())
    print("=" * 72)
    keep = collect.require_complete(df, report)      # raises on a broken/empty matrix

    cells = aggregate.aggregate_seeds(keep)
    tasks = sorted(cells["task"].unique())
    scopes = {"pooled": cells}
    if len(tasks) > 1:
        for task in tasks:
            scopes[task] = cells[cells["task"] == task]

    summary = {
        "results_dir": str(args.results_dir),
        "phase": args.phase,
        "baseline": args.baseline,
        "alpha": args.alpha,
        "method": args.method,
        "tasks": tasks,
        "scopes": {},
    }

    # Per-metric, collect each scope's mean ranks for the cross-task figure.
    per_task_ranks = {m: {} for m in METRICS}

    for scope, scope_cells in scopes.items():
        summary["scopes"][scope] = {}
        for metric in METRICS:
            # Rule A of analysis_plan_amendment_01: a block where no arm beat chance
            # contributes the rank vector (3,3,3,3,3), which adds nothing to the Friedman
            # numerator while still raising the block count, and so lowers power on the
            # blocks that do carry an effect. Accuracy only -- efficiency is measured
            # whether or not the model learned. The exclusion is always reported, because
            # an unstated exclusion is indistinguishable from data that never existed.
            metric_cells, dropped = aggregate.drop_degenerate_blocks(scope_cells, metric)
            if dropped:
                print(f"[{scope}/{metric}] excluded {len(dropped)} degenerate block(s) "
                      f"(no arm above chance + {aggregate.DEGENERATE_MARGIN}): "
                      + ", ".join(f"{t}/d{d:03d}" for t, d in dropped))
            if metric_cells.empty:
                print(f"[{scope}/{metric}] every block was degenerate; metric skipped")
                summary["scopes"][scope][metric] = {
                    "skipped": "all blocks degenerate",
                    "degenerate_blocks": [[t, int(d)] for t, d in dropped],
                }
                continue

            res = analyse_scope(metric_cells, metric, args.baseline, args.alpha,
                                args.method, args.permutations, args.bootstrap)
            summary["scopes"][scope][metric] = _scope_summary(metric, res)
            summary["scopes"][scope][metric]["degenerate_blocks"] = [
                [t, int(d)] for t, d in dropped
            ]

            if res["posthoc"] is not None:
                fig = out / f"cd_{scope}_{metric}.{args.figure_format}"
                plots.critical_difference_diagram(
                    res["mean_ranks"], res["posthoc"],
                    f"{scope}: {metric}", fig,
                )
                table = plots.results_table(res["omnibus"], res["posthoc"], res["effects"])
                table.to_csv(out / f"table_{scope}_{metric}.csv", index=False)

            if scope in tasks:
                per_task_ranks[metric][scope] = res["mean_ranks"]

    # Cross-task figures: one per metric, showing whether the ranking moves.
    if len(tasks) > 1:
        for metric in METRICS:
            if len(per_task_ranks[metric]) > 1:
                plots.task_comparison_figure(
                    per_task_ranks[metric], metric,
                    f"Cross-task ranking: {metric}",
                    out / f"crosstask_{metric}.{args.figure_format}",
                )

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote report to {out}/")
    print(f"  {len(list(out.glob('cd_*')))} critical-difference diagrams")
    print(f"  {len(list(out.glob('crosstask_*')))} cross-task figures")
    print(f"  {len(list(out.glob('table_*.csv')))} results tables")
    print(f"  summary.json")

    # A short console digest of the headline: which arms beat the baseline where.
    print("\nheadline (arms significantly better/worse than "
          f"{args.baseline}, {args.method}-corrected):")
    for scope in scopes:
        for metric in METRICS:
            s = summary["scopes"][scope][metric]
            if "significant_vs_baseline" in s and s["significant_vs_baseline"]:
                best = min(s["mean_ranks"], key=s["mean_ranks"].get)
                print(f"  {scope:>8} / {metric:<28} "
                      f"best={best}  differs from {args.baseline}: "
                      f"{s['significant_vs_baseline']}")


if __name__ == "__main__":
    main()
