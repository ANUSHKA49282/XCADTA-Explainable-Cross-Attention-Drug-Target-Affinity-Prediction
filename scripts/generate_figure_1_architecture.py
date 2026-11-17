from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "final_figures"
OUTPUT_FILE = OUTPUT_DIR / "Figure_1_XCADTA_Architecture.png"


def add_box(ax, x, y, width, height, title, subtitle, color):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5,
        edgecolor=color,
        facecolor=color + "18",
    )
    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#172B4D",
    )

    ax.text(
        x + width / 2,
        y + height * 0.32,
        subtitle,
        ha="center",
        va="center",
        fontsize=8,
        color="#344563",
        wrap=True,
    )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.6,
        color="#52677F",
    )
    ax.add_patch(arrow)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        8,
        8.55,
        "XCADTA: Drug–Target Affinity Prediction Architecture",
        ha="center",
        fontsize=19,
        fontweight="bold",
        color="#142B49",
    )

    ax.text(
        8,
        8.12,
        "Molecular graph encoding, protein sequence representation and bidirectional cross-attention",
        ha="center",
        fontsize=10,
        color="#52677F",
    )

    drug_color = "#2878B5"
    protein_color = "#249B83"
    fusion_color = "#8056A8"
    output_color = "#D17A32"

    # Drug representation branch
    ax.text(
        3.5,
        7.35,
        "DRUG REPRESENTATION",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=drug_color,
    )

    add_box(
        ax, 0.5, 5.95, 2.3, 0.9,
        "Drug SMILES",
        "Chemical string",
        drug_color,
    )

    add_box(
        ax, 3.2, 5.95, 2.3, 0.9,
        "Molecular Graph",
        "65-dimensional atom features",
        drug_color,
    )

    add_box(
        ax, 5.9, 5.95, 2.3, 0.9,
        "Drug Encoder",
        "GAT + GCN layers",
        drug_color,
    )

    add_arrow(ax, (2.8, 6.4), (3.2, 6.4))
    add_arrow(ax, (5.5, 6.4), (5.9, 6.4))

    # Protein representation branch
    ax.text(
        3.5,
        4.95,
        "PROTEIN REPRESENTATION",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=protein_color,
    )

    add_box(
        ax, 0.5, 3.55, 2.3, 0.9,
        "Protein Sequence",
        "Amino-acid sequence",
        protein_color,
    )

    add_box(
        ax, 3.2, 3.55, 2.3, 0.9,
        "ESM-2 Embeddings",
        "Precomputed 1280D features",
        protein_color,
    )

    add_box(
        ax, 5.9, 3.55, 2.3, 0.9,
        "Protein Projection",
        "128-dimensional representation",
        protein_color,
    )

    add_arrow(ax, (2.8, 4.0), (3.2, 4.0))
    add_arrow(ax, (5.5, 4.0), (5.9, 4.0))

    # Fusion
    add_box(
        ax, 9.0, 5.05, 2.6, 1.15,
        "Bidirectional",
        "Drug-to-protein and\nprotein-to-drug attention",
        fusion_color,
    )

    add_arrow(ax, (8.2, 6.4), (9.0, 5.85))
    add_arrow(ax, (8.2, 4.0), (9.0, 5.35))

    add_box(
        ax, 12.1, 5.05, 2.5, 1.15,
        "Fusion Pooling",
        "Masked mean and max pooling\nFour 128D vectors",
        fusion_color,
    )

    add_arrow(ax, (11.6, 5.62), (12.1, 5.62))

    # Regression output
    add_box(
        ax, 12.1, 2.85, 2.5, 1.15,
        "Regression Head",
        "512 → 256 → 64 → 1",
        output_color,
    )

    add_arrow(ax, (13.35, 5.05), (13.35, 4.0))

    add_box(
        ax, 12.1, 0.9, 2.5, 1.05,
        "Predicted Affinity",
        "KIBA regression score",
        output_color,
    )

    add_arrow(ax, (13.35, 2.85), (13.35, 1.95))

    # Architecture notes
    ax.text(
        0.65,
        1.65,
        "Model configuration",
        fontsize=10,
        fontweight="bold",
        color="#142B49",
    )

    ax.text(
        0.65,
        1.25,
        "Drug features: 65D  |  Protein embeddings: 1280D  |  Model dimension: 128D",
        fontsize=9,
        color="#52677F",
    )

    ax.text(
        0.65,
        0.85,
        "Cross-attention: 4 heads  |  Fusion representation: 512D",
        fontsize=9,
        color="#52677F",
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("Figure 1 generated successfully!")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
