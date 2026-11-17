from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = ROOT / "results" / "case_studies" / "candidate_residue_attention_new.csv"
OUTPUT_DIR = ROOT / "results" / "final_figures"
OUTPUT_FILE = OUTPUT_DIR / "Figure_3_Attention_Heatmap.png"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_FILE)

    required = {
        "sequence_position",
        "amino_acid",
        "attention_importance",
        "attention_rank",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values("sequence_position").reset_index(drop=True)

    values = df["attention_importance"].to_numpy(dtype=float)
    positions = df["sequence_position"].to_numpy(dtype=int)

    if not np.isfinite(values).all():
        raise ValueError("Invalid attention values detected.")

    if np.any(values < 0):
        raise ValueError("Logarithmic scaling requires non-negative values.")

    positive = values[values > 0]
    if len(positive) == 0:
        raise ValueError("No positive attention values found.")

    vmin = max(float(positive.min()), 1e-8)
    vmax = float(values.max())

    fig, ax = plt.subplots(figsize=(17, 5.8))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "XCADTA Residue-Level Attention Distribution",
        fontsize=18,
        fontweight="bold",
        color="#183153",
        y=0.97,
    )

    fig.text(
        0.5,
        0.905,
        f"Candidate protein sequence | {len(df)} residue positions",
        ha="center",
        fontsize=10,
        color="#52677F",
    )

    masked_values = np.ma.masked_less_equal(values, 0)

    heatmap = ax.imshow(
        masked_values[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap="inferno",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    ax.set_yticks([0])
    ax.set_yticklabels(["Residue attention"], fontsize=9)

    tick_indices = np.linspace(
        0, len(df) - 1, min(13, len(df)), dtype=int
    )

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(positions[tick_indices], fontsize=8)

    ax.set_xlabel("Protein sequence position", fontsize=10, labelpad=9)

    ax.set_title(
        "Attention importance across all sequence positions",
        fontsize=11,
        fontweight="bold",
        pad=15,
    )

    # Highlight the top 10 residues.
    top = df.nsmallest(10, "attention_rank")

    for row in top.itertuples():
        index = int(
            np.where(positions == int(row.sequence_position))[0][0]
        )

        ax.scatter(
            index,
            0,
            marker="o",
            s=48,
            facecolors="none",
            edgecolors="white",
            linewidths=1.5,
            zorder=4,
        )

    colorbar = fig.colorbar(
        heatmap,
        ax=ax,
        orientation="vertical",
        fraction=0.025,
        pad=0.025,
    )

    colorbar.set_label(
        "Attention importance (log scale)",
        fontsize=9,
        labelpad=10,
    )

    top_labels = "   |   ".join(
        f"{row.amino_acid}{int(row.sequence_position)}"
        for row in top.itertuples()
    )

    fig.text(
        0.5,
        0.17,
        "TOP 10 ATTENDED RESIDUES",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#183153",
    )

    fig.text(
        0.5,
        0.115,
        top_labels,
        ha="center",
        fontsize=9,
        color="#344563",
    )

    fig.text(
        0.5,
        0.045,
        "Logarithmic colour scaling improves visibility of lower values; "
        "original attention values are unchanged. Attention is model-internal "
        "and does not establish experimentally validated binding contacts.",
        ha="center",
        fontsize=8,
        color="#596579",
    )

    plt.tight_layout(rect=[0.04, 0.23, 0.96, 0.86])

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("Polished Figure 3 generated successfully!")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Residues plotted: {len(df)}")
    print(f"Original minimum: {values.min():.8f}")
    print(f"Original maximum: {values.max():.8f}")
    print(f"Top residue: {top.iloc[0]['amino_acid']}"
          f"{int(top.iloc[0]['sequence_position'])}")


if __name__ == "__main__":
    main()
