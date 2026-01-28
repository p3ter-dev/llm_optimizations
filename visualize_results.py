"""
Visualization and Analysis Script

Generates plots and insights for the technical report:
1. Accuracy vs Model Size
2. Accuracy vs Inference Time
3. Trade-off Analysis
4. Pareto Frontier
"""

# MUST be first
import matplotlib

matplotlib.use("Agg")

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def load_results(filename="experiment_results.json"):
    """Load experiment results"""
    with open(filename, "r") as f:
        results = json.load(f)
    return pd.DataFrame(results)


def plot_accuracy_vs_size(df, save_path="accuracy_vs_size.png"):
    """
    Plot Accuracy vs Model Size
    Shows the trade-off between model compression and accuracy
    """
    plt.figure(figsize=(10, 6))

    # Create scatter plot
    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink"]

    for i, (_, row) in enumerate(df.iterrows()):
        plt.scatter(
            row["model_size_mb"],
            row["accuracy"],
            s=200,
            alpha=0.7,
            color=colors[i % len(colors)],
            label=row["model_name"],
        )
        plt.annotate(
            row["model_name"],
            (row["model_size_mb"], row["accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    plt.xlabel("Model Size (MB)", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Accuracy vs Model Size Trade-off", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_accuracy_vs_time(df, save_path="accuracy_vs_time.png"):
    """
    Plot Accuracy vs Inference Time
    Shows the trade-off between speed and accuracy
    """
    plt.figure(figsize=(10, 6))

    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink"]

    for i, (_, row) in enumerate(df.iterrows()):
        plt.scatter(
            row["inference_time"],
            row["accuracy"],
            s=200,
            alpha=0.7,
            color=colors[i % len(colors)],
            label=row["model_name"],
        )
        plt.annotate(
            row["model_name"],
            (row["inference_time"], row["accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    plt.xlabel("Inference Time (seconds)", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Accuracy vs Inference Time Trade-off", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_pareto_frontier(df, save_path="pareto_frontier.png"):
    """
    Plot Pareto Frontier
    Shows optimal models in the accuracy-size-speed space
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink"]

    # Plot 1: Accuracy vs Size
    for i, (_, row) in enumerate(df.iterrows()):
        ax1.scatter(
            row["model_size_mb"],
            row["accuracy"],
            s=300,
            alpha=0.7,
            color=colors[i % len(colors)],
        )
        ax1.annotate(
            row["model_name"],
            (row["model_size_mb"], row["accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    ax1.set_xlabel("Model Size (MB)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Accuracy vs Size", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy vs Time
    for i, (_, row) in enumerate(df.iterrows()):
        ax2.scatter(
            row["inference_time"],
            row["accuracy"],
            s=300,
            alpha=0.7,
            color=colors[i % len(colors)],
        )
        ax2.annotate(
            row["model_name"],
            (row["inference_time"], row["accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    ax2.set_xlabel("Inference Time (seconds)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Accuracy vs Speed", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_comparison_bars(df, save_path="comparison_bars.png"):
    """
    Bar chart comparing all metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = df["model_name"].values

    # Normalize metrics for better visualization
    baseline = df[df["model_name"] == "Baseline"].iloc[0]

    # Plot 1: Accuracy
    ax = axes[0, 0]
    accuracies = df["accuracy"].values
    bars = ax.bar(range(len(models)), accuracies, color="steelblue", alpha=0.7)
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Model Accuracy Comparison", fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.axhline(
        y=baseline["accuracy"], color="red", linestyle="--", label="Baseline", alpha=0.5
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Model Size
    ax = axes[0, 1]
    sizes = df["model_size_mb"].values
    bars = ax.bar(range(len(models)), sizes, color="coral", alpha=0.7)
    ax.set_ylabel("Model Size (MB)", fontweight="bold")
    ax.set_title("Model Size Comparison", fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.axhline(
        y=baseline["model_size_mb"],
        color="red",
        linestyle="--",
        label="Baseline",
        alpha=0.5,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Plot 3: Inference Time
    ax = axes[1, 0]
    times = df["inference_time"].values
    bars = ax.bar(range(len(models)), times, color="lightgreen", alpha=0.7)
    ax.set_ylabel("Inference Time (s)", fontweight="bold")
    ax.set_title("Inference Time Comparison", fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.axhline(
        y=baseline["inference_time"],
        color="red",
        linestyle="--",
        label="Baseline",
        alpha=0.5,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Plot 4: Parameters
    ax = axes[1, 1]
    params = df["num_parameters"].values / 1e6  # Convert to millions
    bars = ax.bar(range(len(models)), params, color="plum", alpha=0.7)
    ax.set_ylabel("Parameters (millions)", fontweight="bold")
    ax.set_title("Parameter Count Comparison", fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.axhline(
        y=baseline["num_parameters"] / 1e6,
        color="red",
        linestyle="--",
        label="Baseline",
        alpha=0.5,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_normalized_comparison(df, save_path="normalized_comparison.png"):
    """
    Radar/Spider chart showing normalized metrics
    """
    # Get baseline for normalization
    baseline = df[df["model_name"] == "Baseline"].iloc[0]

    # Prepare data
    models = df["model_name"].values
    metrics = ["Accuracy", "Size\nEfficiency", "Speed"]

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink"]

    for i, (_, row) in enumerate(df.iterrows()):
        # Normalize metrics (higher is better for all)
        values = [
            row["accuracy"] / baseline["accuracy"],  # Accuracy ratio
            baseline["model_size_mb"]
            / row["model_size_mb"],  # Inverse size (smaller is better)
            baseline["inference_time"]
            / row["inference_time"],  # Inverse time (faster is better)
        ]
        values += values[:1]  # Complete the circle

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=row["model_name"],
            color=colors[i % len(colors)],
            alpha=0.7,
        )
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 2)
    ax.set_title(
        "Normalized Performance Comparison\n(Baseline = 1.0)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def generate_insights(df):
    """
    Generate textual insights for the report
    """
    baseline = df[df["model_name"] == "Baseline"].iloc[0]

    insights = []
    insights.append("=" * 70)
    insights.append("KEY INSIGHTS FOR TECHNICAL REPORT")
    insights.append("=" * 70 + "\n")

    # Best accuracy
    best_acc = df.loc[df["accuracy"].idxmax()]
    insights.append(f"1. HIGHEST ACCURACY:")
    insights.append(f"   {best_acc['model_name']}: {best_acc['accuracy']:.2f}%")
    insights.append(
        f"   ({best_acc['accuracy'] - baseline['accuracy']:+.2f}% vs baseline)\n"
    )

    # Smallest model
    smallest = df.loc[df["model_size_mb"].idxmin()]
    insights.append(f"2. SMALLEST MODEL:")
    insights.append(f"   {smallest['model_name']}: {smallest['model_size_mb']:.2f} MB")
    reduction = (1 - smallest["model_size_mb"] / baseline["model_size_mb"]) * 100
    insights.append(f"   ({reduction:.1f}% size reduction vs baseline)")
    insights.append(
        f"   Accuracy: {smallest['accuracy']:.2f}% "
        f"({smallest['accuracy'] - baseline['accuracy']:+.2f}% vs baseline)\n"
    )

    # Fastest model
    fastest = df.loc[df["inference_time"].idxmin()]
    insights.append(f"3. FASTEST INFERENCE:")
    insights.append(f"   {fastest['model_name']}: {fastest['inference_time']:.2f}s")
    speedup = baseline["inference_time"] / fastest["inference_time"]
    insights.append(f"   ({speedup:.2f}x speedup vs baseline)")
    insights.append(
        f"   Accuracy: {fastest['accuracy']:.2f}% "
        f"({fastest['accuracy'] - baseline['accuracy']:+.2f}% vs baseline)\n"
    )

    # Best trade-off (efficiency score)
    df["efficiency_score"] = (
        (df["accuracy"] / baseline["accuracy"])
        * (baseline["model_size_mb"] / df["model_size_mb"])
        * (baseline["inference_time"] / df["inference_time"])
    )
    best_tradeoff = df.loc[df["efficiency_score"].idxmax()]
    insights.append(f"4. BEST OVERALL TRADE-OFF:")
    insights.append(f"   {best_tradeoff['model_name']}")
    insights.append(f"   Efficiency Score: {best_tradeoff['efficiency_score']:.2f}")
    insights.append(f"   (combines accuracy retention, size reduction, and speed)\n")

    # MoE analysis
    moe_models = df[df["model_name"].str.contains("MoE")]
    if not moe_models.empty:
        insights.append(f"5. MoE ANALYSIS:")
        for _, row in moe_models.iterrows():
            param_increase = (
                row["num_parameters"] / baseline["num_parameters"] - 1
            ) * 100
            insights.append(f"   {row['model_name']}:")
            insights.append(f"     Parameters: +{param_increase:.1f}% vs baseline")
            insights.append(
                f"     Accuracy: {row['accuracy']:.2f}% "
                f"({row['accuracy'] - baseline['accuracy']:+.2f}%)"
            )
        insights.append("")

    # PTQ analysis
    ptq_models = df[df["model_name"].str.contains("PTQ")]
    if not ptq_models.empty:
        insights.append(f"6. QUANTIZATION (PTQ) ANALYSIS:")
        for _, row in ptq_models.iterrows():
            # Find original model
            orig_name = (
                row["model_name"].replace(" + PTQ", "").replace("PTQ", "").strip()
            )
            if orig_name == "":
                orig_name = "Baseline"
            orig = df[df["model_name"] == orig_name]
            if not orig.empty:
                orig = orig.iloc[0]
                size_reduction = (
                    1 - row["model_size_mb"] / orig["model_size_mb"]
                ) * 100
                acc_drop = row["accuracy"] - orig["accuracy"]
                insights.append(f"   {row['model_name']}:")
                insights.append(f"     Size reduction: {size_reduction:.1f}%")
                insights.append(f"     Accuracy change: {acc_drop:+.2f}%")
        insights.append("")

    # KD analysis
    kd_models = df[df["model_name"].str.contains("KD")]
    if not kd_models.empty:
        insights.append(f"7. KNOWLEDGE DISTILLATION ANALYSIS:")
        for _, row in kd_models.iterrows():
            param_reduction = (
                1 - row["num_parameters"] / baseline["num_parameters"]
            ) * 100
            insights.append(f"   {row['model_name']}:")
            insights.append(f"     Parameter reduction: {param_reduction:.1f}%")
            insights.append(
                f"     Accuracy: {row['accuracy']:.2f}% "
                f"({row['accuracy'] - baseline['accuracy']:+.2f}% vs baseline)"
            )
        insights.append("")

    insights.append("=" * 70)

    # Write to file
    with open("insights.txt", "w") as f:
        f.write("\n".join(insights))

    # Print to console
    for line in insights:
        print(line)

    print("\nInsights saved to 'insights.txt'")


def main():
    """Generate all visualizations and insights"""
    print("=" * 70)
    print("GENERATING VISUALIZATIONS AND ANALYSIS")
    print("=" * 70 + "\n")

    # Load results
    try:
        df = load_results("experiment_results.json")
    except FileNotFoundError:
        print("Error: experiment_results.json not found!")
        print("Please run 'python run_all_experiments.py' first.")
        return

    print(f"Loaded {len(df)} experiment results\n")

    # Generate plots
    print("Generating plots...")
    plot_accuracy_vs_size(df)
    plot_accuracy_vs_time(df)
    plot_pareto_frontier(df)
    plot_comparison_bars(df)
    plot_normalized_comparison(df)

    print("\nGenerating insights...")
    generate_insights(df)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - accuracy_vs_size.png")
    print("  - accuracy_vs_time.png")
    print("  - pareto_frontier.png")
    print("  - comparison_bars.png")
    print("  - normalized_comparison.png")
    print("  - insights.txt")


if __name__ == "__main__":
    main()
