#!/usr/bin/env python3
"""
Statistical analysis script to reproduce paper results
Performs t-test analysis comparing Muon vs AdamW grokking epochs
"""

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


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
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    # Filter out experiments that didn't achieve grokking
    grokking_results = df[df["grokking_epoch"].notna()].copy()

    if len(grokking_results) == 0:
        return {"error": "No grokking detected in any experiments"}

    # Separate by optimizer
    muon_results = grokking_results[
        grokking_results["optimizer_type"] == "muon"
    ]
    adamw_results = grokking_results[
        grokking_results["optimizer_type"] == "adamw"
    ]

    if len(muon_results) == 0 or len(adamw_results) == 0:
        return {"error": "Need both Muon and AdamW results for comparison"}

    # Extract grokking epochs
    muon_epochs = muon_results["grokking_epoch"].values
    adamw_epochs = adamw_results["grokking_epoch"].values

    # Calculate statistics
    muon_mean = np.mean(muon_epochs)
    adamw_mean = np.mean(adamw_epochs)
    muon_std = np.std(muon_epochs, ddof=1)
    adamw_std = np.std(adamw_epochs, ddof=1)

    # Perform two-sample t-test
    t_stat, p_value = stats.ttest_ind(
        muon_epochs, adamw_epochs, equal_var=False
    )

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        (
            (len(muon_epochs) - 1) * muon_std**2
            + (len(adamw_epochs) - 1) * adamw_std**2
        )
        / (len(muon_epochs) + len(adamw_epochs) - 2)
    )
    cohens_d = (adamw_mean - muon_mean) / pooled_std

    # Calculate speedup
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
            "significant": p_value < 0.05,
        },
    }


def print_analysis_results(analysis: dict[str, Any]) -> None:
    """Print analysis results in a formatted way"""
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return

    print("=" * 60)
    print("GROKKING ANALYSIS RESULTS")
    print("=" * 60)

    muon_stats = analysis["muon_stats"]
    adamw_stats = analysis["adamw_stats"]
    comparison = analysis["comparison"]

    print("\nMuon Optimizer:")
    print(f"  Mean grokking epoch: {muon_stats['mean_grokking_epoch']:.2f}")
    print(f"  Std grokking epoch:  {muon_stats['std_grokking_epoch']:.2f}")
    print(f"  N experiments:      {muon_stats['n_experiments']}")

    print("\nAdamW Optimizer:")
    print(f"  Mean grokking epoch: {adamw_stats['mean_grokking_epoch']:.2f}")
    print(f"  Std grokking epoch:  {adamw_stats['std_grokking_epoch']:.2f}")
    print(f"  N experiments:      {adamw_stats['n_experiments']}")

    print("\nStatistical Comparison:")
    print(f"  T-statistic: {comparison['t_statistic']:.4f}")
    print(f"  P-value:     {comparison['p_value']:.2e}")
    print(f"  Cohen's d:   {comparison['cohens_d']:.4f}")
    print(f"  Speedup:     {comparison['speedup']:.2f}x")
    print(f"  Significant: {'Yes' if comparison['significant'] else 'No'}")

    # Compare with paper results
    print("\nComparison with Paper Results:")
    paper_muon_mean = 102.89
    paper_adamw_mean = 153.09
    paper_t_stat = 5.0175
    paper_p_value = 6.33e-08

    print(f"  Paper Muon mean:    {paper_muon_mean:.2f}")
    print(f"  Our Muon mean:      {muon_stats['mean_grokking_epoch']:.2f}")
    print(f"  Paper AdamW mean:   {paper_adamw_mean:.2f}")
    print(f"  Our AdamW mean:     {adamw_stats['mean_grokking_epoch']:.2f}")
    print(f"  Paper t-statistic:  {paper_t_stat:.4f}")
    print(f"  Our t-statistic:    {comparison['t_statistic']:.4f}")
    print(f"  Paper p-value:      {paper_p_value:.2e}")
    print(f"  Our p-value:        {comparison['p_value']:.2e}")


def analyze_by_task_and_softmax(results: list[dict[str, Any]]) -> None:
    """Analyze results broken down by task and softmax variant"""
    df = pd.DataFrame(results)
    grokking_results = df[df["grokking_epoch"].notna()].copy()

    if len(grokking_results) == 0:
        print("No grokking detected in any experiments")
        return

    print("\n" + "=" * 60)
    print("BREAKDOWN BY TASK AND SOFTMAX VARIANT")
    print("=" * 60)

    for task in grokking_results["task_type"].unique():
        print(f"\nTask: {task}")
        task_results = grokking_results[grokking_results["task_type"] == task]

        for softmax in task_results["softmax_variant"].unique():
            print(f"  Softmax: {softmax}")
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

                print(f"    Muon mean:   {muon_mean:.2f}")
                print(f"    AdamW mean:  {adamw_mean:.2f}")
                print(f"    Speedup:    {speedup:.2f}x")
            else:
                print("    Missing data for one optimizer")


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

    # Load results
    results = load_results(args.results_file)

    # Perform analysis
    analysis = analyze_grokking_results(results)

    # Print results
    print_analysis_results(analysis)

    if args.detailed:
        analyze_by_task_and_softmax(results)

    # Save analysis if requested
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to {args.output_file}")


if __name__ == "__main__":
    main()
