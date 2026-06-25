"""Generate comparison plots: base vs LoRA on both tasks + cross-domain heatmap.

Produces:
  - judge_comparison.png: grouped bar chart (Accuracy, Precision, Recall, F1)
  - poet_comparison.png: grouped bar chart (Format, Rhyme, Topic)
  - cross_domain_heatmap.png: 3x2 heatmap with YlOrRd colormap

Usage:
  python -m eval.plot_results [--output-dir DIR]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# macOS Chinese font configuration
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = ['Songti SC', 'STHeiti', 'PingFang HK', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BASE_COLOR = '#3498db'
LORA_COLOR = '#e74c3c'
POET_LORA_COLOR = '#2ecc71'


# ---------------------------------------------------------------------------
# Judge comparison bar chart
# ---------------------------------------------------------------------------
def plot_judge_comparison(output_dir: str) -> None:
    """Grouped bar chart: Judge task -- Base Model vs Judge LoRA.

    Shows Accuracy, Precision, Recall, F1 with delta labels.
    """
    base_path = os.path.join(output_dir, "judge_base.json")
    lora_path = os.path.join(output_dir, "judge_lora.json")

    if not os.path.exists(base_path) or not os.path.exists(lora_path):
        print("  Missing judge eval files. Skipping judge comparison plot.")
        return

    with open(base_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(lora_path, "r", encoding="utf-8") as f:
        lora = json.load(f)

    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision\n(UPDATE)", "Recall\n(UPDATE)", "F1\n(UPDATE)"]

    base_vals = [base.get(m, 0) for m in metrics]
    lora_vals = [lora.get(m, 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35

    bars_base = ax.bar(x - width / 2, base_vals, width,
                       label="Base Model", color=BASE_COLOR, edgecolor='white', linewidth=1)
    bars_lora = ax.bar(x + width / 2, lora_vals, width,
                       label="Judge LoRA", color=LORA_COLOR, edgecolor='white', linewidth=1)

    # Value labels on bars
    for bar in bars_base:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    for bar in bars_lora:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Delta labels
    for i, (b, l) in enumerate(zip(base_vals, lora_vals)):
        delta = l - b
        ax.annotate(f'Delta={delta:+.3f}',
                    xy=(x[i], max(b, l) + 0.05),
                    ha='center', va='bottom', fontsize=9, color='red')

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Judge Task: Base Model vs Judge LoRA", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "judge_comparison.png"), dpi=150)
    print(f"  Saved: {output_dir}/judge_comparison.png")
    plt.close()


# ---------------------------------------------------------------------------
# Poet comparison bar chart
# ---------------------------------------------------------------------------
def plot_poet_comparison(output_dir: str) -> None:
    """Grouped bar chart: Poet task -- Base Model vs Poet LoRA.

    Shows format_compliance, rhyme_compliance, topic_relevance with delta labels.
    """
    base_path = os.path.join(output_dir, "poet_base.json")
    lora_path = os.path.join(output_dir, "poet_lora.json")

    if not os.path.exists(base_path) or not os.path.exists(lora_path):
        print("  Missing poet eval files. Skipping poet comparison plot.")
        return

    with open(base_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(lora_path, "r", encoding="utf-8") as f:
        lora = json.load(f)

    metrics = ["format_compliance", "rhyme_compliance", "topic_relevance"]
    labels = ["Format\nCompliance", "Rhyme\nCompliance", "Topic\nRelevance"]

    base_vals = [base.get(m, 0) for m in metrics]
    lora_vals = [lora.get(m, 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35

    bars_base = ax.bar(x - width / 2, base_vals, width,
                       label="Base Model", color=BASE_COLOR, edgecolor='white', linewidth=1)
    bars_lora = ax.bar(x + width / 2, lora_vals, width,
                       label="Poet LoRA", color=POET_LORA_COLOR, edgecolor='white', linewidth=1)

    # Value labels on bars
    for bar in bars_base:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    for bar in bars_lora:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Delta labels
    for i, (b, l) in enumerate(zip(base_vals, lora_vals)):
        delta = l - b
        ax.annotate(f'Delta={delta:+.3f}',
                    xy=(x[i], max(b, l) + 0.05),
                    ha='center', va='bottom', fontsize=9, color='green')

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Poet Task: Base Model vs Poet LoRA", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "poet_comparison.png"), dpi=150)
    print(f"  Saved: {output_dir}/poet_comparison.png")
    plt.close()


# ---------------------------------------------------------------------------
# Cross-domain heatmap
# ---------------------------------------------------------------------------
def plot_cross_domain_heatmap(output_dir: str) -> None:
    """3x2 heatmap showing each model's performance on each task.

    Uses YlOrRd colormap with colored borders highlighting specialization cells.
    """
    cross_path = os.path.join(output_dir, "cross_eval.json")

    if not os.path.exists(cross_path):
        print("  Missing cross_eval.json. Skipping cross-domain heatmap.")
        return

    with open(cross_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build the 3x2 matrix
    models_labels = ["Base Model", "Judge LoRA", "Poet LoRA"]
    tasks_labels = ["Judge (Accuracy)", "Poet (Format)"]

    # Judge task scores (accuracy), row order: base, judge, poet
    judge_scores = [
        data.get("base_judge", {}).get("accuracy", 0),
        data.get("judge_judge", {}).get("accuracy", 0),
        data.get("poet_judge", {}).get("accuracy", 0),
    ]

    # Poet task scores (form compliance), row order: base, poet, judge
    poet_scores = [
        data.get("base_poet", {}).get("avg_form_compliance", 0),
        data.get("poet_poet", {}).get("avg_form_compliance", 0),
        data.get("judge_poet", {}).get("avg_form_compliance", 0),
    ]

    matrix = np.array([judge_scores, poet_scores]).T  # 3x2

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)

    # Axis labels
    ax.set_xticks(np.arange(len(tasks_labels)))
    ax.set_yticks(np.arange(len(models_labels)))
    ax.set_xticklabels(tasks_labels, fontsize=11)
    ax.set_yticklabels(models_labels, fontsize=11)

    # Text annotations in each cell
    for i in range(len(models_labels)):
        for j in range(len(tasks_labels)):
            val = matrix[i, j]
            text_color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center", color=text_color,
                    fontsize=12, fontweight='bold')

    # Colored borders for specialization cells
    # Judge LoRA -> Judge Task (row 1, col 0): should be high
    ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False, edgecolor='red', linewidth=3))
    # Poet LoRA -> Poet Task (row 2, col 1): should be high
    ax.add_patch(plt.Rectangle((1, 2), 1, 1, fill=False, edgecolor='green', linewidth=3))

    ax.set_title("Cross-Domain Evaluation: Model x Task", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label="Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_domain_heatmap.png"), dpi=150)
    print(f"  Saved: {output_dir}/cross_domain_heatmap.png")
    plt.close()


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------
def generate_all_plots(output_dir: str) -> None:
    """Call all 3 plot functions."""
    print("Generating evaluation result plots...")
    plot_judge_comparison(output_dir)
    plot_poet_comparison(output_dir)
    plot_cross_domain_heatmap(output_dir)
    print("Done! All plots saved to:", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation result plots",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory containing eval JSONs and for output plots (default: <project>/results)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )

    os.makedirs(output_dir, exist_ok=True)
    generate_all_plots(output_dir)


if __name__ == "__main__":
    main()