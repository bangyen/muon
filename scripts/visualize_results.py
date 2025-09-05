import argparse
import json
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Suppress scipy precision warnings for statistical tests
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="scipy.stats"
)


def create_grokking_comparison_plot(
    df: pd.DataFrame, output_dir: str = "plots"
):
    """
    Create comparison plot showing grokking epochs for Muon vs AdamW
    Matches Figure 4 from the paper
    """
    os.makedirs(output_dir, exist_ok=True)

    grokking_df = df[df["grokking_epoch"].notna()].copy()

    if len(grokking_df) == 0:
        print("No grokking detected in any experiments")
        return None

    plt.figure(figsize=(10, 6))

    sns.boxplot(data=grokking_df, x="optimizer_type", y="grokking_epoch")
    plt.title(
        "Distribution of Grokking Epochs for Muon and AdamW Optimizers",
        fontsize=14,
    )
    plt.ylabel("Grokking Epoch", fontsize=12)
    plt.xlabel("Optimizer", fontsize=12)

    muon_mean = grokking_df[grokking_df["optimizer_type"] == "muon"][
        "grokking_epoch"
    ].mean()
    adamw_mean = grokking_df[grokking_df["optimizer_type"] == "adamw"][
        "grokking_epoch"
    ].mean()

    plt.axhline(
        y=muon_mean,
        xmin=0,
        xmax=0.5,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"Muon mean: {muon_mean:.1f}",
    )
    plt.axhline(
        y=adamw_mean,
        xmin=0.5,
        xmax=1,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"AdamW mean: {adamw_mean:.1f}",
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "grokking_comparison_boxplot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=grokking_df, x="optimizer_type", y="grokking_epoch")
    plt.title("Distribution of Grokking Epochs (Violin Plot)", fontsize=14)
    plt.ylabel("Grokking Epoch", fontsize=12)
    plt.xlabel("Optimizer", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "grokking_comparison_violin.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

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
        print(f"Muon average grokking epoch: {muon_grokking.mean():.2f}")
        print(f"AdamW average grokking epoch: {adamw_grokking.mean():.2f}")
        print(f"Speedup: {adamw_grokking.mean() / muon_grokking.mean():.2f}x")

        from scipy import stats

        t_stat, p_value = stats.ttest_ind(
            muon_grokking, adamw_grokking, equal_var=False
        )
        print(f"T-test: t={t_stat:.4f}, p={p_value:.2e}")

        print("\nComparison with Paper Results:")
        print("Paper Muon mean:    102.89")
        print(f"Our Muon mean:      {muon_grokking.mean():.2f}")
        print("Paper AdamW mean:   153.09")
        print(f"Our AdamW mean:     {adamw_grokking.mean():.2f}")
        print("Paper t-statistic:  5.0175")
        print(f"Our t-statistic:    {t_stat:.4f}")
        print("Paper p-value:     6.33e-08")
        print(f"Our p-value:        {p_value:.2e}")

    return grokking_df


def create_learning_curves_plot(
    results: list[dict], output_dir: str = "plots"
):
    """
    Create learning curves showing training and validation accuracy over time
    """
    os.makedirs(output_dir, exist_ok=True)

    sample_results = []
    for result in results:
        if result["grokking_epoch"] is not None:
            sample_results.append(result)

    if len(sample_results) == 0:
        print("No successful grokking experiments to plot")
        return

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

        for result in task_results[:3]:
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

    grokking_df = df[df["grokking_epoch"].notna()].copy()

    if len(grokking_df) == 0:
        print("No grokking detected for ablation study")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

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

    ax3 = axes[1, 0]
    heatmap_data = (
        grokking_df.groupby(["task_type", "softmax_variant"])["grokking_epoch"]
        .mean()
        .unstack()
    )
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="viridis", ax=ax3)
    ax3.set_title("Grokking Epochs: Task vs Softmax Variant")

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

    summary_stats = []

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

    summary_df = pd.DataFrame(summary_stats)

    summary_file = os.path.join(output_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_file, index=False)

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

    print(f"Loading results from {args.results_file}")
    results = load_results(args.results_file)

    df = pd.DataFrame(results)

    print(f"Loaded {len(results)} experiment results")
    print(f"DataFrame shape: {df.shape}")

    print("\nCreating visualizations...")

    print("Creating grokking comparison plot...")
    create_grokking_comparison_plot(df, args.output_dir)

    print("Creating learning curves...")
    create_learning_curves_plot(results, args.output_dir)

    print("Creating ablation study plots...")
    create_ablation_study_plot(df, args.output_dir)

    print("Creating summary table...")
    summary_df = create_summary_table(df, args.output_dir)

    print(f"\nAll visualizations saved to {args.output_dir}")
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
