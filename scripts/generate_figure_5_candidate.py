from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "case_studies" / "candidate_residue_attention_new.csv"
OUTPUT_DIR = ROOT / "results" / "final_figures"
OUTPUT_FILE = OUTPUT_DIR / "Figure_5_Candidate_Residue_Attention.png"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "attention_rank",
        "attention_importance",
        "amino_acid",
        "sequence_position",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    df["attention_rank"] = pd.to_numeric(
        df["attention_rank"], errors="raise"
    )
    df["attention_importance"] = pd.to_numeric(
        df["attention_importance"], errors="raise"
    )

    if df.empty:
        raise ValueError("The residue attention CSV is empty.")

    top = (
        df.sort_values("attention_rank")
        .head(15)
        .copy()
    )

    top["residue_label"] = (
        top["amino_acid"].astype(str)
        + top["sequence_position"].astype(int).astype(str)
    )

    top = top.sort_values("attention_importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(
        top["residue_label"],
        top["attention_importance"],
        height=0.68,
    )

    ax.set_title(
        "Top 15 Protein Residues by XCADTA Attention",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    ax.set_xlabel("Attention Importance", fontsize=12)
    ax.set_ylabel("Protein Residue (one-based position)", fontsize=12)

    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, top["attention_importance"]):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.4f}",
            va="center",
            fontsize=9,
        )

    fig.text(
        0.5,
        0.015,
        "Attention represents model-internal signals, not experimentally "
        "validated binding-site residues. Drug and target identities remain unverified.",
        ha="center",
        fontsize=8,
        wrap=True,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Figure 5 generated successfully!")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Residues plotted: {len(top)}")


if __name__ == "__main__":
    main()
