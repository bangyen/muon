#!/usr/bin/env python3
"""
Display CI smoke test results in a readable format.

This script is used by the CI workflow to display smoke test results
in a clean, readable format without YAML parsing issues.
"""

import json
from pathlib import Path


def display_results():
    """Display smoke test results from JSON and CSV files."""
    results_dir = Path("results")

    if not results_dir.exists():
        print("No results directory found")
        return

    # Display CSV summary
    csv_files = list(results_dir.glob("experiment_summary_*.csv"))
    if csv_files:
        print("CSV Summary:")
        with open(csv_files[0]) as f:
            print(f.read())

    # Display JSON results
    json_files = list(results_dir.glob("experiment_results_*.json"))
    if json_files:
        print("JSON Results:")
        with open(json_files[0]) as f:
            data = json.load(f)

        for result in data:
            print(
                f"Task: {result['task_type']}, "
                f"Optimizer: {result['optimizer_type']}, "
                f"Grokking Epoch: {result['grokking_epoch']}, "
                f"Final Val Acc: {result['final_val_acc']:.3f}"
            )


if __name__ == "__main__":
    display_results()
