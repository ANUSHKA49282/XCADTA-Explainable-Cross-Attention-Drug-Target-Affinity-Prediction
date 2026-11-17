
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.data import Batch


# Atom feature positions from data_utils.py
_FEATURE_GROUPS: Tuple[Tuple[str, int, int], ...] = (
    ("atom_type", 0, 44),
    ("degree", 44, 51),
    ("formal_charge", 51, 52),
    ("hybridization", 52, 58),
    ("aromaticity", 58, 59),
    ("total_hydrogens", 59, 64),
    ("chirality", 64, 65),
)


# Calculate Spearman correlation safely.
def _safe_spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None

    result = spearmanr(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float)
    )

    value = float(
        result.statistic if hasattr(result, "statistic") else result[0]
    )

    return value if math.isfinite(value) else None


# Get the prediction for one drug-protein pair.
def _prediction_for_graph(
    model,
    graph,
    protein_emb: torch.Tensor,
    device: torch.device
) -> float:

    graph_batch = Batch.from_data_list([graph.clone()]).to(device)
    protein_batch = protein_emb.unsqueeze(0).to(device)

    mask = torch.ones(
        (1, protein_emb.shape[0]),
        dtype=torch.bool,
        device=device
    )

    with torch.no_grad():
        prediction, _ = model(graph_batch, protein_batch, mask)

    return float(prediction.reshape(-1)[0].detach().cpu().item())


# Get predictions for multiple protein embedding variants.
def _predict_protein_variants(
    model,
    graph,
    variants: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 8
) -> np.ndarray:

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if not variants:
        return np.zeros(0, dtype=np.float64)

    outputs: List[float] = []

    for start in range(0, len(variants), batch_size):
        chunk = variants[start:start + batch_size]

        graph_batch = Batch.from_data_list(
            [graph.clone() for _ in chunk]
        ).to(device)

        protein_batch = torch.stack(chunk, dim=0).to(device)

        protein_mask = torch.ones(
            (len(chunk), protein_batch.shape[1]),
            dtype=torch.bool,
            device=device
        )

        with torch.no_grad():
            prediction, _ = model(
                graph_batch,
                protein_batch,
                protein_mask
            )

        outputs.extend(
            prediction.detach().reshape(-1).cpu().tolist()
        )

    return np.asarray(outputs, dtype=np.float64)


# Calculate atom-level Integrated Gradients.
def _atom_integrated_gradients(
    model,
    graph,
    protein_emb: torch.Tensor,
    molecule,
    device: torch.device,
    steps: int = 64
) -> Dict:

    if steps < 2:
        raise ValueError("Integrated Gradients requires at least 2 steps.")

    if graph.x.shape[0] != molecule.GetNumAtoms():
        raise ValueError(
            "Molecule atom count does not match the molecular graph."
        )

    x = graph.x.detach().to(
        device=device,
        dtype=torch.float32
    )

    baseline = torch.zeros_like(x)

    protein_batch = protein_emb.unsqueeze(0).to(device)

    protein_mask = torch.ones(
        (1, protein_emb.shape[0]),
        dtype=torch.bool,
        device=device
    )

    alphas = torch.linspace(
        0.0,
        1.0,
        steps,
        device=device
    )

    gradient_sum = torch.zeros_like(x)

    model.eval()

    with torch.enable_grad():
        for index, alpha in enumerate(alphas):

            x_step = (
                baseline + alpha * (x - baseline)
            ).detach().requires_grad_(True)

            graph_step = graph.clone()
            graph_step.x = x_step

            graph_batch = Batch.from_data_list(
                [graph_step]
            ).to(device)

            prediction, _ = model(
                graph_batch,
                protein_batch,
                protein_mask
            )

            gradient = torch.autograd.grad(
                prediction.sum(),
                x_step,
                retain_graph=False
            )[0]

            weight = (
                0.5
                if index in (0, len(alphas) - 1)
                else 1.0
            )

            gradient_sum += weight * gradient

    average_gradient = gradient_sum / (steps - 1)

    attributions = (
        (x - baseline) * average_gradient
    ).detach().cpu().numpy()

    actual_prediction = _prediction_for_graph(
        model, graph, protein_emb, device
    )

    baseline_graph = graph.clone()
    baseline_graph.x = baseline.detach().cpu()

    baseline_prediction = _prediction_for_graph(
        model, baseline_graph, protein_emb, device
    )

    completeness_residual = abs(
        float(attributions.sum())
        - (actual_prediction - baseline_prediction)
    )

    completeness_scale = max(
        abs(actual_prediction - baseline_prediction),
        abs(float(attributions.sum())),
        1e-8
    )

    completeness_error = (
        completeness_residual / completeness_scale
    )

    atom_rows = []

    for atom_index in range(attributions.shape[0]):

        atom = molecule.GetAtomWithIdx(atom_index)
        feature_values = attributions[atom_index]

        group_contributions = {}

        for group_name, start, end in _FEATURE_GROUPS:
            if start < len(feature_values):
                group_contributions[group_name] = float(
                    feature_values[
                        start:min(end, len(feature_values))
                    ].sum()
                )

        if len(feature_values) != 65:
            group_contributions = {
                "all_atom_features": float(feature_values.sum())
            }

        signed_contribution = float(feature_values.sum())
        importance = float(np.abs(feature_values).sum())

        strongest_group = (
            max(
                group_contributions,
                key=lambda key: abs(group_contributions[key])
            )
            if group_contributions
            else "unknown"
        )

        if signed_contribution > 0:
            direction = "raises model output"
        elif signed_contribution < 0:
            direction = "lowers model output"
        else:
            direction = "neutral"

        atom_rows.append({
            "atom_index": atom_index,
            "element": atom.GetSymbol(),
            "is_aromatic": bool(atom.GetIsAromatic()),
            "formal_charge": int(atom.GetFormalCharge()),
            "contribution": signed_contribution,
            "importance": importance,
            "direction": direction,
            "strongest_feature_group": strongest_group,
            "feature_contributions": group_contributions,
        })

    ranked_atoms = sorted(
        atom_rows,
        key=lambda row: row["importance"],
        reverse=True
    )

    for rank, row in enumerate(ranked_atoms, start=1):
        row["rank"] = rank

    return {
        "method": "Integrated Gradients",
        "baseline": "zero-valued atom-feature reference",
        "steps": steps,
        "prediction": actual_prediction,
        "baseline_prediction": baseline_prediction,
        "completeness_error": completeness_error,
        "completeness_residual": completeness_residual,
        "atoms": sorted(
            ranked_atoms,
            key=lambda row: row["atom_index"]
        ),
        "ranked_atoms": ranked_atoms,
        "feature_groups": (
            [name for name, _, _ in _FEATURE_GROUPS]
            if attributions.shape[1] == 65
            else ["all_atom_features"]
        ),
        "interpretation": (
            "Positive contributions raise the model output relative "
            "to the zero-feature reference; negative contributions "
            "lower it. Importance is the sum of absolute feature "
            "attributions for each atom."
        ),
    }


# Estimate residue importance using sampled feature-order permutations.
def _protein_permutation_shap(
    model,
    graph,
    protein_emb: torch.Tensor,
    candidate_indices: List[int],
    baseline_embedding: torch.Tensor,
    device: torch.device,
    permutations: int = 128,
    seed: int = 2026
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    rng = np.random.default_rng(seed)

    candidate_count = len(candidate_indices)

    if candidate_count == 0:
        return (
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            0.0
        )

    if permutations < 2:
        raise ValueError("At least 2 permutations are required.")

    base_state = protein_emb.clone()

    for residue_index in candidate_indices:
        base_state[residue_index] = baseline_embedding

    baseline_prediction = float(
        _predict_protein_variants(
            model,
            graph,
            [base_state],
            device
        )[0]
    )

    all_contributions = np.zeros(
        (permutations, candidate_count),
        dtype=np.float64
    )

    for permutation_index in range(permutations):

        order = rng.permutation(candidate_count)

        state = base_state.clone()
        states = [state.clone()]

        for candidate_slot in order:

            residue_index = candidate_indices[
                int(candidate_slot)
            ]

            state = state.clone()
            state[residue_index] = protein_emb[residue_index]

            states.append(state)

        predictions = _predict_protein_variants(
            model,
            graph,
            states,
            device
        )

        marginal_contributions = np.diff(predictions)

        for step_index, candidate_slot in enumerate(order):
            all_contributions[
                permutation_index,
                int(candidate_slot)
            ] = marginal_contributions[step_index]

    shap_values = all_contributions.mean(axis=0)

    midpoint = permutations // 2

    first_half = all_contributions[:midpoint].mean(axis=0)
    second_half = all_contributions[midpoint:].mean(axis=0)

    return (
        shap_values,
        first_half,
        second_half,
        baseline_prediction
    )


# Compare top-ranked residues with randomly selected residues.
def _candidate_conditioned_validation(
    model,
    graph,
    protein_emb: torch.Tensor,
    candidate_indices: List[int],
    baseline_embedding: torch.Tensor,
    ranked_candidate_indices: List[int],
    original_prediction: float,
    device: torch.device,
    random_repeats: int = 20,
    seed: int = 99
) -> Dict:

    if not candidate_indices:
        raise ValueError("At least one candidate residue is required.")

    if random_repeats < 1:
        raise ValueError("random_repeats must be at least 1.")

    candidate_set = set(candidate_indices)

    fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    def variant_for(masked_indices: Sequence[int]) -> torch.Tensor:

        variant = protein_emb.clone()

        for idx in masked_indices:
            if idx in candidate_set:
                variant[idx] = baseline_embedding

        return variant

    top_variants = []

    for fraction in fractions:

        count = int(
            round(fraction * len(ranked_candidate_indices))
        )

        top_variants.append(
            variant_for(ranked_candidate_indices[:count])
        )

    top_predictions = _predict_protein_variants(
        model,
        graph,
        top_variants,
        device
    )

    top_effects = np.abs(
        top_predictions - original_prediction
    )

    top_auc = float(
        np.trapezoid(top_effects, fractions)
    )

    rng = np.random.default_rng(seed)

    random_effect_curves = []

    for _ in range(random_repeats):

        random_order = list(
            np.asarray(candidate_indices)[
                rng.permutation(len(candidate_indices))
            ]
        )

        random_variants = []

        for fraction in fractions:

            count = int(
                round(fraction * len(random_order))
            )

            random_variants.append(
                variant_for(random_order[:count])
            )

        random_predictions = _predict_protein_variants(
            model,
            graph,
            random_variants,
            device
        )

        random_effect_curves.append(
            np.abs(random_predictions - original_prediction)
        )

    random_effects = np.mean(
        np.stack(random_effect_curves, axis=0),
        axis=0
    )

    random_auc = float(
        np.trapezoid(random_effects, fractions)
    )

    sufficiency = []

    keep_counts = sorted(set([
        min(5, len(ranked_candidate_indices)),
        min(10, len(ranked_candidate_indices))
    ]))

    for keep_count in keep_counts:

        keep = set(
            ranked_candidate_indices[:keep_count]
        )

        masked = [
            idx
            for idx in candidate_indices
            if idx not in keep
        ]

        prediction = float(
            _predict_protein_variants(
                model,
                graph,
                [variant_for(masked)],
                device
            )[0]
        )

        sufficiency.append({
            "kept_top_residues": keep_count,
            "prediction": prediction,
            "absolute_prediction_difference": abs(
                prediction - original_prediction
            ),
        })

    return {
        "fractions_removed": fractions,
        "top_shap_absolute_prediction_change": [
            float(value) for value in top_effects
        ],
        "random_absolute_prediction_change": [
            float(value) for value in random_effects
        ],
        "top_shap_deletion_auc": top_auc,
        "random_deletion_auc": random_auc,
        "deletion_lift_over_random": (
            float(top_auc / random_auc)
            if random_auc > 1e-12
            else None
        ),
        "sufficiency": sufficiency,
        "scope": (
            "Only the attention-screened candidate residues are "
            "perturbed; all other residues remain at their "
            "observed embeddings."
        ),
    }


# Generate atom and protein explanations for one prediction.
def generate_explanation(
    model,
    graph,
    protein_emb: torch.Tensor,
    sequence: str,
    molecule,
    device: torch.device,
    max_residues: int = 20,
    permutations: int = 128,
    ig_steps: int = 64
) -> Dict:

    model.eval()

    protein_emb = torch.as_tensor(
        protein_emb,
        dtype=torch.float32,
        device=device
    )

    if protein_emb.ndim != 2:
        raise ValueError(
            "Protein embeddings must have shape (L, D)."
        )

    if protein_emb.shape[0] == 0:
        raise ValueError("Protein embedding sequence is empty.")

    if protein_emb.shape[1] == 0:
        raise ValueError("Protein embedding dimension is empty.")

    if protein_emb.shape[0] != len(sequence):
        raise ValueError(
            "Protein embedding length does not match the supplied "
            "sequence length. Check whether the ESM cache includes "
            "special tokens."
        )

    if not torch.isfinite(protein_emb).all():
        raise ValueError("Protein embeddings contain invalid values.")

    if graph.x.shape[0] != molecule.GetNumAtoms():
        raise ValueError(
            "Molecule atom count does not match the molecular graph."
        )

    if graph.x.shape[1] != 65:
        raise ValueError(
            f"Expected 65 atom features, received {graph.x.shape[1]}."
        )

    if max_residues <= 0:
        raise ValueError("max_residues must be greater than zero.")

    if permutations < 4:
        raise ValueError("At least 4 permutations are required.")

    if ig_steps < 2:
        raise ValueError("Integrated Gradients requires at least 2 steps.")

    if graph.x.shape[0] == 0:
        raise ValueError("Molecular graph contains no atoms.")

    graph_batch = Batch.from_data_list(
        [graph.clone()]
    ).to(device)

    protein_mask = torch.ones(
        (1, protein_emb.shape[0]),
        dtype=torch.bool,
        device=device
    )

    with torch.no_grad():

        original_prediction_tensor, attention_tensor = model(
            graph_batch,
            protein_emb.unsqueeze(0),
            protein_mask
        )

    original_prediction = float(
        original_prediction_tensor.reshape(-1)[0].cpu().item()
    )

    if attention_tensor is None:
        raise ValueError("The model did not return attention values.")

    attention_matrix = attention_tensor[0].detach().cpu().numpy()

    if attention_matrix.ndim != 2:
        raise ValueError(
            "Expected attention with shape (atoms, residues)."
        )

    if attention_matrix.shape[0] != graph.x.shape[0]:
        raise ValueError(
            "Attention atom dimension does not match the molecular graph."
        )

    if attention_matrix.shape[1] != len(sequence):
        raise ValueError(
            "Attention residue dimension does not match the protein sequence."
        )

    if not np.all(np.isfinite(attention_matrix)):
        raise ValueError("Attention contains invalid values.")

    # Average attention across all drug atoms.
    mean_attention = attention_matrix.mean(axis=0)

    candidate_count = min(
        max_residues,
        len(sequence)
    )

    candidate_indices = np.argsort(
        -mean_attention
    )[:candidate_count].astype(int).tolist()

    candidate_indices.sort()

    baseline_embedding = protein_emb.mean(dim=0)

    shap_values, first_half, second_half, shap_baseline_prediction = (
        _protein_permutation_shap(
            model=model,
            graph=graph,
            protein_emb=protein_emb,
            candidate_indices=candidate_indices,
            baseline_embedding=baseline_embedding,
            device=device,
            permutations=permutations
        )
    )

    ranked_slots = np.argsort(-np.abs(shap_values))

    ranked_candidate_indices = [
        candidate_indices[int(slot)]
        for slot in ranked_slots
    ]

    residue_rows = []

    for slot, residue_index in enumerate(candidate_indices):

        value = float(shap_values[slot])

        if value > 0:
            direction = "raises model output"
        elif value < 0:
            direction = "lowers model output"
        else:
            direction = "neutral"

        residue_rows.append({
            "position": residue_index + 1,
            "amino_acid": sequence[residue_index],
            "shap_value": value,
            "importance": abs(value),
            "attention": float(mean_attention[residue_index]),
            "direction": direction,
        })

    residue_rows.sort(
        key=lambda row: row["importance"],
        reverse=True
    )

    additivity_error = abs(
        (original_prediction - shap_baseline_prediction)
        - float(shap_values.sum())
    )

    stability = _safe_spearman(
        np.abs(first_half),
        np.abs(second_half)
    )

    candidate_attention = [
        float(mean_attention[idx])
        for idx in candidate_indices
    ]

    candidate_abs_shap = [
        float(abs(value))
        for value in shap_values
    ]

    attention_shap_agreement = _safe_spearman(
        candidate_attention,
        candidate_abs_shap
    )

    validation = _candidate_conditioned_validation(
        model=model,
        graph=graph,
        protein_emb=protein_emb,
        candidate_indices=candidate_indices,
        baseline_embedding=baseline_embedding,
        ranked_candidate_indices=ranked_candidate_indices,
        original_prediction=original_prediction,
        device=device
    )

    atom_explanation = _atom_integrated_gradients(
        model=model,
        graph=graph,
        protein_emb=protein_emb,
        molecule=molecule,
        device=device,
        steps=ig_steps
    )

    return {
        "status": "complete",
        "prediction": original_prediction,

        "protein_residue_explanation": {
            "method": "Attention-screened Monte-Carlo permutation SHAP",
            "baseline": (
                "Each candidate residue embedding is replaced by "
                "the mean embedding of the supplied protein sequence; "
                "non-candidate residues remain observed."
            ),
            "candidate_selection": (
                f"Top {candidate_count} residues by mean "
                "drug-to-protein attention, averaged across real drug atoms."
            ),
            "permutations": permutations,
            "stability_method": (
                "Spearman correlation between absolute SHAP rankings "
                "estimated from the first and second halves of sampled permutations."
            ),
            "candidate_residues": residue_rows,
            "top_residues": residue_rows[:min(20, len(residue_rows))],
            "baseline_prediction": shap_baseline_prediction,
            "additivity_error": float(additivity_error),
            "attention_shap_spearman": attention_shap_agreement,
            "stability_spearman_split_half": stability,
            "validation": validation,
            "interpretation": (
                "Positive SHAP values raise the predicted model output "
                "relative to the stated embedding baseline; negative "
                "values lower it. These are model attributions, not "
                "experimentally verified binding effects."
            ),
        },

        "drug_atom_explanation": atom_explanation,

        "metrics": {
            "shap_reconstruction_residual": float(additivity_error),
            "shap_additivity_error": float(additivity_error),
            "deletion_auc_top_shap": validation["top_shap_deletion_auc"],
            "deletion_auc_random": validation["random_deletion_auc"],
            "deletion_lift_over_random": validation["deletion_lift_over_random"],
            "sufficiency_top_residue_tests": validation["sufficiency"],
            "attribution_stability_spearman": stability,
            "attention_shap_spearman": attention_shap_agreement,
            "integrated_gradients_completeness_error": (
                atom_explanation["completeness_error"]
            ),
            "integrated_gradients_completeness_residual": (
                atom_explanation["completeness_residual"]
            ),
        },

        "methodological_notes": [
            (
                "Residue SHAP is estimated with sampled feature-order "
                "permutations over an attention-screened candidate set."
            ),
            (
                "Deletion and sufficiency tests use the same candidate "
                "set and mean-embedding reference; they are not external "
                "biological validation."
            ),
            (
                "Integrated Gradients uses a zero-valued atom-feature "
                "reference and trapezoidal integration."
            ),
            (
                "Cross-attention indicates model attention allocation "
                "and must not be interpreted as a confirmed physical contact map."
            ),
        ],
    }
