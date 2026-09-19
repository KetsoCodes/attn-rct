"""Executes the Demsar analysis pipeline.

Transforms result JSONs into a complete report (CD diagrams, cross-task plots, 
and summary tables). Refuses to analyze incomplete or broken data matrices.

Usage:
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
    """Runs the statistical pipeline (omnibus, effects, post-hoc) for one metric."""
    direction = aggregate.METRIC_DIRECTION[metric]
    ranked = aggregate.rank_within_blocks(cells, metric)
    rank_matrix = aggregate.rank_matrix(ranked)
    mean_ranks = aggregate.mean_ranks(ranked)

    om = omnibus.run_omnibus(
        rank_matrix, metric, alpha=alpha, n_permutations=n_permutations
    )

    result = {"omnibus": om, "mean_ranks": mean_ranks, "posthoc": None, "effects": None}
    
    # Always compute effects; non-rejection is not definitive equivalence.
    result["effects"] = effects.compute_effects(
        aggregate.value_matrix(cells, metric), 
        baseline, 
        metric, 
        direction,
        n_bootstrap=n_bootstrap, 
        alpha=alpha,
    )
    
    # Post-hoc requires omnibus rejection.
    if om.reject:
        result["posthoc"] = posthoc.compare_to_control(
            mean_ranks, 
            baseline, 
            len(rank_matrix), 
            metric, 
            method=method, 
            alpha=alpha,
        )
        
    return result


def _scope_summary(metric, res) -> dict:
    """Formats a metric's analysis results for JSON serialization."""
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

    # 1. Load and rigidly validate matrix integrity.
    df, report = collect.load_results(args.results_dir, phase=args.phase)
    print(report.summary())
    print("=" * 72)
    keep = collect.require_complete(df, report)

    # 2. Aggregate seeds and define scopes.
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

    # Track ranks for cross-task comparison figures.
    per_task_ranks = {m: {} for m in METRICS}

    # 3. Execute analysis pipeline per scope and metric.
    for scope, scope_cells in scopes.items():
        summary["scopes"][scope] = {}
        
        for metric in METRICS:
            res = analyse_scope(
                scope_cells, metric, args.baseline, args.alpha,
                args.method, args.permutations, args.bootstrap
            )
            summary["scopes"][scope][metric] = _scope_summary(metric, res)

            # Generate artifacts if post-hoc tests ran.
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

    # 4. Generate cross-task figures.
    if len(tasks) > 1:
        for metric in METRICS:
            if len(per_task_ranks[metric]) > 1:
                plots.task_comparison_figure(
                    per_task_ranks[metric], metric,
                    f"Cross-task ranking: {metric}",
                    out / f"crosstask_{metric}.{args.figure_format}",
                )

    # 5. Output summary JSON and console digest.
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote report to {out}/")
    print(f"  {len(list(out.glob('cd_*')))} critical-difference diagrams")
    print(f"  {len(list(out.glob('crosstask_*')))} cross-task figures")
    print(f"  {len(list(out.glob('table_*.csv')))} results tables")
    print(f"  summary.json")

    print(f"\nheadline (arms significantly better/worse than {args.baseline}, "
          f"{args.method}-corrected):")
          
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