
#Compares model attention with known binding-pocket residues for selected kinase-inhibitor pairs. This analysis is qualitative and does not claim experimentalor biological validation.


from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def validate_inputs(
    attn_residue_importance: np.ndarray,
    known_pocket_residues: List[int],
    k: int,
):
    """Check attention values, residue indices and k."""

    attention = np.asarray(attn_residue_importance, dtype=float)

    if attention.ndim != 1:
        raise ValueError("Attention importance must be a 1D array.")

    if len(attention) == 0:
        raise ValueError("Attention array is empty.")

    if not np.all(np.isfinite(attention)):
        raise ValueError("Attention contains invalid values.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    known = set(known_pocket_residues)

    for index in known:
        if not isinstance(index, (int, np.integer)):
            raise ValueError("Pocket residue indices must be integers.")

        if index < 0 or index >= len(attention):
            raise ValueError(
                f"Pocket residue index {index} is outside the "
                f"attention array (length {len(attention)})."
            )

    return attention, known


def top_k_recall(
    attn_residue_importance: np.ndarray,
    known_pocket_residues: List[int],
    k: int = 10,
) -> float:
    """
    Calculate the fraction of known pocket residues
    present among the top-k attention residues.

    Residue indices must be zero-based Python array indices.
    Convert PDB residue numbering before calling this function.
    """

    attention, known = validate_inputs(
        attn_residue_importance,
        known_pocket_residues,
        k,
    )

    if not known:
        return float("nan")

    actual_k = min(k, len(attention))

    top_k_indices = set(
        np.argsort(-attention)[:actual_k].tolist()
    )

    return len(top_k_indices & known) / len(known)


def plot_attention_pocket_overlay(
    attn_residue_importance: np.ndarray,
    known_pocket_residues: List[int],
    title: str,
    out_path: str,
):
    """Plot residue attention and highlight known pocket residues."""

    attention, known = validate_inputs(
        attn_residue_importance,
        known_pocket_residues,
        k=1,
    )

    colors = [
        "#d62728" if i in known else "#1f77b4"
        for i in range(len(attention))
    ]

    fig, ax = plt.subplots(figsize=(12, 3.5))

    ax.bar(
        range(len(attention)),
        attention,
        color=colors,
        width=1.0,
    )

    ax.set_xlabel("Residue index (zero-based)")
    ax.set_ylabel("Attention weight (summed over atoms)")
    ax.set_title(title)

    if known:
        ax.legend(handles=[
            plt.Rectangle(
                (0, 0), 1, 1,
                color="#d62728",
                label="Known pocket residue",
            ),
            plt.Rectangle(
                (0, 0), 1, 1,
                color="#1f77b4",
                label="Other residue",
            ),
        ])
    else:
        ax.text(
            0.5,
            0.95,
            "Known pocket residues not provided",
            transform=ax.transAxes,
            ha="center",
            va="top",
            color="red",
        )

    fig.tight_layout()

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Pocket residue lists must be filled only after
# verifying the exact protein sequence and residue numbering.

CASE_STUDY_TEMPLATE: Dict[str, dict] = {
    "ABL1_imatinib": {
        "known_pocket_residues": [],
        "note": "Verify residue numbering against the exact protein sequence.",
    },
    "EGFR_erlotinib": {
        "known_pocket_residues": [],
        "note": "Verify residue numbering against the exact protein sequence.",
    },
}


def run_case_study(
    pair_name: str,
    attn_residue_importance: np.ndarray,
    known_pocket_residues: List[int],
    k: int = 10,
    out_dir: str = "results/case_studies",
) -> Dict[str, object]:

    recall = top_k_recall(
        attn_residue_importance,
        known_pocket_residues,
        k=k,
    )

    out_path = Path(out_dir) / f"{pair_name}_overlay.png"

    plot_attention_pocket_overlay(
        attn_residue_importance,
        known_pocket_residues,
        title=f"{pair_name}: attention vs known pocket",
        out_path=str(out_path),
    )

    if np.isnan(recall):
        verdict = "not evaluated"
        print(
            f"[{pair_name}] Pocket comparison not evaluated: "
            "no known pocket residues supplied."
        )
    else:
        verdict = "yes" if recall > 0 else "no"

        print(
            f"[{pair_name}] top-{k} recall = {recall:.2f} | "
            f"falls within pocket: {verdict}"
        )

    return {
        "pair": pair_name,
        "top_k_recall": recall,
        "verdict": verdict,
        "figure": str(out_path),
    }
