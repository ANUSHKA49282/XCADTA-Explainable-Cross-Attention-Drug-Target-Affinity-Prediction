from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT_DIR = RESULTS / "final_figures"
OUTPUT_FILE = OUTPUT_DIR / "Figure_2_XCADTA_vs_Concat_Ablation.png"

METRICS = {
    "MSE": {
        "label": "Mean Squared Error (MSE)",
        "better": "lower",
    },
    "CI": {
        "label": "Concordance Index (CI)",
        "better": "higher",
    },
    "r2m": {
        "label": "Modified R² (r2m)",
        "better": "higher",
    },
}


def load_metrics(filename, model_key):
    path = RESULTS / filename

    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if model_key not in data:
        raise KeyError(
            f"Expected key '{model_key}' was not found in {filename}"
        )

    values = data[model_key]

    for metric in METRICS:
        if metric not in values:
            raise KeyError(
                f"Metric '{metric}' missing from {filename}"
            )

    return {metric: float(values[metric]) for metric in METRICS}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xcadta = load_metrics(
        "metrics_cleaned_test.json",
        "xcadta",
    )

    concat = load_metrics(
        "metrics_concat_ablation.json",
        "concat_ablation",
    )

    models = ["XCADTA", "Concat Ablation"]
    results = [xcadta, concat]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "XCADTA vs. Concat Ablation",
        fontsize=18,
        fontweight="bold",
        color="#183153",
        y=0.97,
    )

    fig.text(
        0.5,
        0.91,
        "Performance comparison using the selected evaluation metrics",
        ha="center",
        fontsize=10,
        color="#52677F",
    )

    positions = np.arange(2)

    for ax, metric, details in zip(axes, METRICS, METRICS.values()):
        values = [result[metric] for result in results]

        bars = ax.bar(
            positions,
            values,
            width=0.55,
            edgecolor="#FFFFFF",
            linewidth=1.2,
        )

        ax.set_title(
            details["label"],
            fontsize=11,
            fontweight="bold",
            color="#183153",
            pad=14,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel("Score", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        max_value = max(values)
        ax.set_ylim(0, max_value * 1.23 if max_value > 0 else 1)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_value * 0.035,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#26364A",
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(
        0.5,
        0.055,
        "Sources: metrics_cleaned_test.json and metrics_concat_ablation.json. "
        "Confirm matching evaluation splits and protocols before interpreting "
        "this as a controlled ablation comparison.",
        ha="center",
        fontsize=8,
        color="#596579",
        wrap=True,
    )

    plt.tight_layout(rect=[0.03, 0.13, 0.97, 0.87])

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("Figure 2 generated successfully!")
    print(f"Output: {OUTPUT_FILE}")
    print("\nValues plotted:")

    for metric in METRICS:
        print(
            f"{metric}: XCADTA = {xcadta[metric]:.6f} | "
            f"Concat Ablation = {concat[metric]:.6f}"
        )


if __name__ == "__main__":
    main()
