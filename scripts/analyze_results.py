#!/usr/bin/env python3
"""
Statistical analysis script to reproduce paper results
Performs t-test analysis comparing Muon vs AdamW grokking epochs
"""

import argparse
import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

# Suppress scipy precision warnings for statistical tests
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="scipy.stats"
)


def load_results(results_file: str) -> list[dict[str, Any]]:
    """Load experiment results from JSON file"""
    with open(results_file) as f:
        return json.load(f)


def analyze_grokking_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Perform statistical analysis as described in the paper

    Args:
        results: List of experiment results

    Returns:
        Dictionary containing statistical analysis results
    """
    df = pd.DataFrame(results)

    grokking_results = df[df["grokking_epoch"].notna()].copy()

    if len(grokking_results) == 0:
        return {"error": "No grokking detected in any experiments"}

    muon_results = grokking_results[
        grokking_results["optimizer_type"] == "muon"
    ]
    adamw_results = grokking_results[
        grokking_results["optimizer_type"] == "adamw"
    ]

    if len(muon_results) == 0 or len(adamw_results) == 0:
        return {"error": "Need both Muon and AdamW results for comparison"}

    muon_epochs = muon_results["grokking_epoch"].values
    adamw_epochs = adamw_results["grokking_epoch"].values

    muon_mean = np.mean(muon_epochs)
    adamw_mean = np.mean(adamw_epochs)
    muon_std = np.std(muon_epochs, ddof=1)
    adamw_std = np.std(adamw_epochs, ddof=1)

    t_stat, p_value = stats.ttest_ind(
        muon_epochs, adamw_epochs, equal_var=False
    )

    pooled_std = np.sqrt(
        (
            (len(muon_epochs) - 1) * muon_std**2
            + (len(adamw_epochs) - 1) * adamw_std**2
        )
        / (len(muon_epochs) + len(adamw_epochs) - 2)
    )
    cohens_d = (adamw_mean - muon_mean) / pooled_std

    speedup = adamw_mean / muon_mean if muon_mean > 0 else float("inf")

    return {
        "muon_stats": {
            "mean_grokking_epoch": muon_mean,
            "std_grokking_epoch": muon_std,
            "n_experiments": len(muon_epochs),
            "epochs": muon_epochs.tolist(),
        },
        "adamw_stats": {
            "mean_grokking_epoch": adamw_mean,
            "std_grokking_epoch": adamw_std,
            "n_experiments": len(adamw_epochs),
            "epochs": adamw_epochs.tolist(),
        },
        "comparison": {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "speedup": speedup,
            "significant": bool(p_value < 0.05),
        },
    }


def print_analysis_results(analysis: dict[str, Any]) -> None:
    """Print analysis results in a formatted way using tabulate"""
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return

    print("=" * 60)
    print("GROKKING ANALYSIS RESULTS")
    print("=" * 60)

    muon_stats = analysis["muon_stats"]
    adamw_stats = analysis["adamw_stats"]
    comparison = analysis["comparison"]

    # Create optimizer comparison table
    optimizer_data = [
        [
            "Muon Optimizer",
            f"{muon_stats['mean_grokking_epoch']:.2f}",
            f"{muon_stats['std_grokking_epoch']:.2f}",
            muon_stats["n_experiments"],
        ],
        [
            "AdamW Optimizer",
            f"{adamw_stats['mean_grokking_epoch']:.2f}",
            f"{adamw_stats['std_grokking_epoch']:.2f}",
            adamw_stats["n_experiments"],
        ],
    ]

    optimizer_headers = [
        "Optimizer",
        "Mean Grokking Epoch",
        "Std Grokking Epoch",
        "N Experiments",
    ]
    print("\nOptimizer Comparison:")
    print(
        tabulate(
            optimizer_data, headers=optimizer_headers, tablefmt="fancy_grid"
        )
    )

    # Create statistical comparison table
    stats_data = [
        ["T-statistic", f"{comparison['t_statistic']:.4f}"],
        ["P-value", f"{comparison['p_value']:.2e}"],
        ["Cohen's d", f"{comparison['cohens_d']:.4f}"],
        ["Speedup", f"{comparison['speedup']:.2f}x"],
        ["Significant", "Yes" if comparison["significant"] else "No"],
    ]

    print("\nStatistical Comparison:")
    print(
        tabulate(
            stats_data, headers=["Metric", "Value"], tablefmt="fancy_grid"
        )
    )

    # Create paper comparison table
    paper_muon_mean = 102.89
    paper_adamw_mean = 153.09
    paper_t_stat = 5.0175
    paper_p_value = 6.33e-08

    comparison_data = [
        [
            "Muon Mean",
            f"{paper_muon_mean:.2f}",
            f"{muon_stats['mean_grokking_epoch']:.2f}",
        ],
        [
            "AdamW Mean",
            f"{paper_adamw_mean:.2f}",
            f"{adamw_stats['mean_grokking_epoch']:.2f}",
        ],
        [
            "T-statistic",
            f"{paper_t_stat:.4f}",
            f"{comparison['t_statistic']:.4f}",
        ],
        ["P-value", f"{paper_p_value:.2e}", f"{comparison['p_value']:.2e}"],
    ]

    print("\nComparison with Paper Results:")
    print(
        tabulate(
            comparison_data,
            headers=["Metric", "Paper", "Our Results"],
            tablefmt="fancy_grid",
        )
    )


def analyze_by_task_and_softmax(results: list[dict[str, Any]]) -> None:
    """Analyze results broken down by task and softmax variant using tabulate"""
    df = pd.DataFrame(results)
    grokking_results = df[df["grokking_epoch"].notna()].copy()

    if len(grokking_results) == 0:
        print("No grokking detected in any experiments")
        return

    print("\n" + "=" * 60)
    print("BREAKDOWN BY TASK AND SOFTMAX VARIANT")
    print("=" * 60)

    breakdown_data = []

    for task in grokking_results["task_type"].unique():
        task_results = grokking_results[grokking_results["task_type"] == task]

        for softmax in task_results["softmax_variant"].unique():
            softmax_results = task_results[
                task_results["softmax_variant"] == softmax
            ]

            muon_results = softmax_results[
                softmax_results["optimizer_type"] == "muon"
            ]
            adamw_results = softmax_results[
                softmax_results["optimizer_type"] == "adamw"
            ]

            if len(muon_results) > 0 and len(adamw_results) > 0:
                muon_mean = muon_results["grokking_epoch"].mean()
                adamw_mean = adamw_results["grokking_epoch"].mean()
                speedup = (
                    adamw_mean / muon_mean if muon_mean > 0 else float("inf")
                )

                breakdown_data.append(
                    [
                        task,
                        softmax,
                        f"{muon_mean:.2f}",
                        f"{adamw_mean:.2f}",
                        f"{speedup:.2f}x",
                    ]
                )
            else:
                breakdown_data.append(
                    [task, softmax, "N/A", "N/A", "Missing data"]
                )

    if breakdown_data:
        headers = [
            "Task",
            "Softmax Variant",
            "Muon Mean",
            "AdamW Mean",
            "Speedup",
        ]
        print(tabulate(breakdown_data, headers=headers, tablefmt="fancy_grid"))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Muon Optimizer grokking experiment results"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to JSON results file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save analysis results (optional)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed breakdown by task and softmax variant",
    )

    args = parser.parse_args()

    results = load_results(args.results_file)

    analysis = analyze_grokking_results(results)

    print_analysis_results(analysis)

    if args.detailed:
        analyze_by_task_and_softmax(results)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to {args.output_file}")


if __name__ == "__main__":
    main()
