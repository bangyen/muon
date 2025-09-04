import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_grokking_comparison_plot(
    df: pd.DataFrame, output_dir: str = "plots"
):
    """
    Create comparison plot showing grokking epochs for Muon vs AdamW
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter for experiments that achieved grokking
    grokking_df = df[df["grokking_epoch"].notna()].copy()

    if len(grokking_df) == 0:
        print("No grokking detected in any experiments")
        return None

    # Create comparison plot
    plt.figure(figsize=(12, 8))

    # Box plot comparison
    plt.subplot(2, 2, 1)
    sns.boxplot(data=grokking_df, x="optimizer_type", y="grokking_epoch")
    plt.title("Grokking Epoch Comparison: Muon vs AdamW")
    plt.ylabel("Grokking Epoch")
    plt.xlabel("Optimizer")

    # Violin plot for distribution
    plt.subplot(2, 2, 2)
    sns.violinplot(data=grokking_df, x="optimizer_type", y="grokking_epoch")
    plt.title("Distribution of Grokking Epochs")
    plt.ylabel("Grokking Epoch")
    plt.xlabel("Optimizer")

    # Task-wise comparison
    plt.subplot(2, 2, 3)
    task_comparison = (
        grokking_df.groupby(["task_type", "optimizer_type"])["grokking_epoch"]
        .mean()
        .unstack()
    )
    task_comparison.plot(kind="bar", ax=plt.gca())
    plt.title("Grokking Epochs by Task")
    plt.ylabel("Average Grokking Epoch")
    plt.xlabel("Task Type")
    plt.xticks(rotation=45)
    plt.legend(title="Optimizer")

    # Softmax variant comparison
    plt.subplot(2, 2, 4)
    softmax_comparison = (
        grokking_df.groupby(["softmax_variant", "optimizer_type"])[
            "grokking_epoch"
        ]
        .mean()
        .unstack()
    )
    softmax_comparison.plot(kind="bar", ax=plt.gca())
    plt.title("Grokking Epochs by Softmax Variant")
    plt.ylabel("Average Grokking Epoch")
    plt.xlabel("Softmax Variant")
    plt.xticks(rotation=45)
    plt.legend(title="Optimizer")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "grokking_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("GROKKING COMPARISON SUMMARY")
    print("=" * 50)

    muon_grokking = grokking_df[grokking_df["optimizer_type"] == "muon"][
        "grokking_epoch"
    ]
    adamw_grokking = grokking_df[grokking_df["optimizer_type"] == "adamw"][
        "grokking_epoch"
    ]

    if len(muon_grokking) > 0 and len(adamw_grokking) > 0:
        print(
            f"Muon average grokking epoch: {muon_grokking.mean():.2f} ± {muon_grokking.std():.2f}"
        )
        print(
            f"AdamW average grokking epoch: {adamw_grokking.mean():.2f} ± {adamw_grokking.std():.2f}"
        )
        print(f"Speedup: {adamw_grokking.mean() / muon_grokking.mean():.2f}x")

        # Statistical test
        from scipy import stats

        t_stat, p_value = stats.ttest_ind(muon_grokking, adamw_grokking)
        print(f"T-test: t={t_stat:.4f}, p={p_value:.2e}")

    return grokking_df


def create_learning_curves_plot(
    results: list[dict], output_dir: str = "plots"
):
    """
    Create learning curves showing training and validation accuracy over time
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select a few representative experiments for visualization
    sample_results = []
    for result in results:
        if (
            result["grokking_epoch"] is not None
        ):  # Only show successful grokking
            sample_results.append(result)

    if len(sample_results) == 0:
        print("No successful grokking experiments to plot")
        return

    # Create subplots for different tasks
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    tasks = ["gcd", "add", "div", "exp", "mul", "parity"]

    for i, task in enumerate(tasks):
        if i >= len(axes):
            break

        ax = axes[i]
        task_results = [r for r in sample_results if r["task_type"] == task]

        if len(task_results) == 0:
            ax.text(
                0.5,
                0.5,
                f"No data for {task}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Task: {task}")
            continue

        # Plot learning curves
        for result in task_results[:3]:  # Show first 3 experiments
            epochs = range(len(result["val_accuracies"]))
            optimizer = result["optimizer_type"]
            color = "blue" if optimizer == "muon" else "red"
            linestyle = "-" if optimizer == "muon" else "--"

            ax.plot(
                epochs,
                result["val_accuracies"],
                color=color,
                linestyle=linestyle,
                alpha=0.7,
                label=f"{optimizer} (grok: {result['grokking_epoch']})",
            )

        ax.set_title(f"Task: {task}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "learning_curves.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_ablation_study_plot(df: pd.DataFrame, output_dir: str = "plots"):
    """
    Create ablation study plots showing the effect of different components
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter for experiments that achieved grokking
    grokking_df = df[df["grokking_epoch"].notna()].copy()

    if len(grokking_df) == 0:
        print("No grokking detected for ablation study")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Optimizer comparison across all tasks
    ax1 = axes[0, 0]
    optimizer_task_comparison = (
        grokking_df.groupby(["task_type", "optimizer_type"])["grokking_epoch"]
        .mean()
        .unstack()
    )
    optimizer_task_comparison.plot(kind="bar", ax=ax1)
    ax1.set_title("Grokking Epochs by Task and Optimizer")
    ax1.set_ylabel("Average Grokking Epoch")
    ax1.set_xlabel("Task Type")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(title="Optimizer")

    # 2. Softmax variant comparison
    ax2 = axes[0, 1]
    softmax_comparison = (
        grokking_df.groupby(["softmax_variant", "optimizer_type"])[
            "grokking_epoch"
        ]
        .mean()
        .unstack()
    )
    softmax_comparison.plot(kind="bar", ax=ax2)
    ax2.set_title("Grokking Epochs by Softmax Variant")
    ax2.set_ylabel("Average Grokking Epoch")
    ax2.set_xlabel("Softmax Variant")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(title="Optimizer")

    # 3. Heatmap of task vs softmax variant
    ax3 = axes[1, 0]
    heatmap_data = (
        grokking_df.groupby(["task_type", "softmax_variant"])["grokking_epoch"]
        .mean()
        .unstack()
    )
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="viridis", ax=ax3)
    ax3.set_title("Grokking Epochs: Task vs Softmax Variant")

    # 4. Distribution comparison
    ax4 = axes[1, 1]
    muon_data = grokking_df[grokking_df["optimizer_type"] == "muon"][
        "grokking_epoch"
    ]
    adamw_data = grokking_df[grokking_df["optimizer_type"] == "adamw"][
        "grokking_epoch"
    ]

    ax4.hist(muon_data, alpha=0.7, label="Muon", bins=15, color="blue")
    ax4.hist(adamw_data, alpha=0.7, label="AdamW", bins=15, color="red")
    ax4.set_title("Distribution of Grokking Epochs")
    ax4.set_xlabel("Grokking Epoch")
    ax4.set_ylabel("Frequency")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "ablation_study.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_summary_table(df: pd.DataFrame, output_dir: str = "plots"):
    """
    Create a summary table of results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create summary statistics
    summary_stats = []

    # Overall statistics
    total_experiments = len(df)
    grokking_experiments = len(df[df["grokking_epoch"].notna()])

    summary_stats.append(
        {"Metric": "Total Experiments", "Value": total_experiments}
    )

    summary_stats.append(
        {"Metric": "Successful Grokking", "Value": grokking_experiments}
    )

    summary_stats.append(
        {
            "Metric": "Grokking Success Rate",
            "Value": f"{grokking_experiments / total_experiments * 100:.1f}%",
        }
    )

    # Optimizer comparison
    for optimizer in ["muon", "adamw"]:
        optimizer_df = df[df["optimizer_type"] == optimizer]
        optimizer_grokking = optimizer_df[
            optimizer_df["grokking_epoch"].notna()
        ]

        if len(optimizer_grokking) > 0:
            avg_grokking = optimizer_grokking["grokking_epoch"].mean()
            std_grokking = optimizer_grokking["grokking_epoch"].std()
            success_rate = len(optimizer_grokking) / len(optimizer_df) * 100

            summary_stats.append(
                {
                    "Metric": f"{optimizer.upper()} Avg Grokking Epoch",
                    "Value": f"{avg_grokking:.1f} ± {std_grokking:.1f}",
                }
            )

            summary_stats.append(
                {
                    "Metric": f"{optimizer.upper()} Success Rate",
                    "Value": f"{success_rate:.1f}%",
                }
            )

    # Create table
    summary_df = pd.DataFrame(summary_stats)

    # Save as CSV
    summary_file = os.path.join(output_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_file, index=False)

    # Create a nice formatted table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    plt.title("Experiment Summary Statistics", fontsize=16, pad=20)
    plt.savefig(
        os.path.join(output_dir, "summary_table.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return summary_df


def load_results(results_file: str) -> list[dict]:
    """Load experiment results from JSON file"""
    with open(results_file) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Create visualizations for Muon Optimizer experiments"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to experiment results JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_file}")
    results = load_results(args.results_file)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    print(f"Loaded {len(results)} experiment results")
    print(f"DataFrame shape: {df.shape}")

    # Create visualizations
    print("\nCreating visualizations...")

    # 1. Grokking comparison plot
    print("Creating grokking comparison plot...")
    create_grokking_comparison_plot(df, args.output_dir)

    # 2. Learning curves
    print("Creating learning curves...")
    create_learning_curves_plot(results, args.output_dir)

    # 3. Ablation study
    print("Creating ablation study plots...")
    create_ablation_study_plot(df, args.output_dir)

    # 4. Summary table
    print("Creating summary table...")
    summary_df = create_summary_table(df, args.output_dir)

    print(f"\nAll visualizations saved to {args.output_dir}")
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
