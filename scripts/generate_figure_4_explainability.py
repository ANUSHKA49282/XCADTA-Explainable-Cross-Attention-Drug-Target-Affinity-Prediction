
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "results" / "final_figures" / "explainability_validation.json"
OUTPUT_PATH = (
    ROOT / "results" / "final_figures"
    / "Figure_4_Explainability_Validation.png"
)

with JSON_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

explanation = data["explanation"]
metrics = explanation["metrics"]
validation = explanation["protein_residue_explanation"]["validation"]

fractions = np.asarray(validation["fractions_removed"], dtype=float)
top_effects = np.asarray(
    validation["top_shap_absolute_prediction_change"], dtype=float
)
random_effects = np.asarray(
    validation["random_absolute_prediction_change"], dtype=float
)

sufficiency = validation["sufficiency"]
keep_counts = [item["kept_top_residues"] for item in sufficiency]
prediction_differences = [
    item["absolute_prediction_difference"] for item in sufficiency
]

# Figure styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
fig.subplots_adjust(
    left=0.09, right=0.94, top=0.78, bottom=0.19, wspace=0.32
)

fig.suptitle(
    "XCADTA Explainability Validation",
    fontsize=17,
    fontweight="bold",
    y=0.96,
)

# Panel A: Deletion test
ax = axes[0]

top_auc = metrics["deletion_auc_top_shap"]
random_auc = metrics["deletion_auc_random"]
lift = metrics["deletion_lift_over_random"]

ax.plot(
    fractions,
    top_effects,
    marker="o",
    linewidth=2.5,
    markersize=6,
    label=f"Top SHAP residues (AUC = {top_auc:.4f})",
)

ax.plot(
    fractions,
    random_effects,
    marker="s",
    linewidth=2.5,
    markersize=6,
    linestyle="--",
    label=f"Random residues (AUC = {random_auc:.4f})",
)

ax.set_title("(A) Residue deletion analysis", fontweight="bold")
ax.set_xlabel("Fraction of candidate residues removed")
ax.set_ylabel("Absolute prediction change")
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax.text(
    0.97,
    0.12,
    f"Deletion lift = {lift:.2f}×",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    fontweight="bold",
)

# Panel B: Sufficiency test
ax = axes[1]

bars = ax.bar(
    [f"Top {count}" for count in keep_counts],
    prediction_differences,
    width=0.52,
)

ax.set_title("(B) Top-residue sufficiency", fontweight="bold")
ax.set_xlabel("Retained top-ranked candidate residues")
ax.set_ylabel("Absolute prediction difference")
ax.set_ylim(0, max(prediction_differences) * 1.45)
ax.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, prediction_differences):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Validation summary
stability = metrics["attribution_stability_spearman"]
agreement = metrics["attention_shap_spearman"]
ig_error = metrics["integrated_gradients_completeness_error"]

summary = (
    f"Attribution stability (Spearman): {stability:.4f}"
    f"     |     Attention–SHAP agreement: {agreement:.4f}"
    f"     |     IG completeness error: {ig_error:.4f}"
)

fig.text(
    0.5,
    0.075,
    summary,
    ha="center",
    fontsize=9,
)

fig.text(
    0.5,
    0.025,
    "Model-level perturbation tests on attention-screened candidate residues; "
    "not experimental binding-site validation.",
    ha="center",
    fontsize=8,
    style="italic",
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("FIGURE 4 GENERATED SUCCESSFULLY")
print(f"Output: {OUTPUT_PATH}")
print(f"Deletion AUC (top SHAP): {top_auc:.4f}")
print(f"Deletion AUC (random): {random_auc:.4f}")
print(f"Deletion lift: {lift:.4f}x")
print(f"Attribution stability: {stability:.4f}")
