"""Combine method result CSVs and plot RMSE/runtime curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_breakdown_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Return the first sparsity level where each method stops beating baseline."""
    rows = []
    for method, group in results.groupby("method"):
        group = group.sort_values("observed_ratings_removed")
        failed = group[group["beats_baseline"] == False]  # noqa: E712
        first_failed = failed.iloc[0] if not failed.empty else None
        best = group.loc[group["rmse"].idxmin()]
        worst = group.loc[group["rmse"].idxmax()]

        rows.append(
            {
                "method": method,
                "breakdown_observed_removed": None
                if first_failed is None
                else first_failed["observed_ratings_removed"],
                "breakdown_label": "not observed"
                if first_failed is None
                else first_failed["label"],
                "best_rmse": best["rmse"],
                "best_label": best["label"],
                "worst_rmse": worst["rmse"],
                "worst_label": worst["label"],
                "rmse_degradation": worst["rmse"] - best["rmse"],
                "max_skip_rate": group["skip_rate"].max()
                if "skip_rate" in group
                else None,
                "max_fallback_rate": group["fallback_rate"].max()
                if "fallback_rate" in group
                else None,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare recommender results.")
    parser.add_argument("csvs", nargs="+", help="Result CSV files to combine.")
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Directory for comparison figures.",
    )
    args = parser.parse_args()

    missing_csvs = [path for path in args.csvs if not Path(path).exists()]
    if missing_csvs:
        missing = "\n".join(f"  - {path}" for path in missing_csvs)
        raise FileNotFoundError(
            "Result CSV file(s) not found:\n"
            f"{missing}\n"
            "Run the method scripts successfully before running compare_results.py."
        )

    frames = [pd.read_csv(path) for path in args.csvs]
    results = pd.concat(frames, ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, group in results.groupby("method"):
        group = group.sort_values("observed_ratings_removed")
        ax.plot(
            group["observed_ratings_removed"] * 100,
            group["rmse"],
            marker="o",
            linewidth=2,
            label=method,
        )
    if "baseline_rmse" in results.columns:
        baseline = (
            results.sort_values("observed_ratings_removed")
            .drop_duplicates("observed_ratings_removed")
            .sort_values("observed_ratings_removed")
        )
        ax.plot(
            baseline["observed_ratings_removed"] * 100,
            baseline["baseline_rmse"],
            marker="x",
            linestyle="--",
            linewidth=2,
            color="black",
            label="Global Mean Baseline",
        )
    ax.set_xlabel("Observed ratings removed (%)")
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("RMSE vs controlled sparsity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_comparison.png", dpi=150)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, group in results.groupby("method"):
        group = group.sort_values("observed_ratings_removed")
        ax.plot(
            group["observed_ratings_removed"] * 100,
            group["total_time"],
            marker="o",
            linewidth=2,
            label=method,
        )
    ax.set_xlabel("Observed ratings removed (%)")
    ax.set_ylabel("Total time (seconds)")
    ax.set_title("Runtime vs controlled sparsity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_comparison.png", dpi=150)

    combined_path = output_dir.parent / "combined_results.csv"
    summary_path = output_dir.parent / "breakdown_summary.csv"
    results.to_csv(combined_path, index=False)
    if {"beats_baseline", "baseline_rmse"}.issubset(results.columns):
        build_breakdown_summary(results).to_csv(summary_path, index=False)
    print(f"Saved {combined_path}")
    if summary_path.exists():
        print(f"Saved {summary_path}")
    print(f"Saved {output_dir / 'rmse_comparison.png'}")
    print(f"Saved {output_dir / 'runtime_comparison.png'}")


if __name__ == "__main__":
    main()
