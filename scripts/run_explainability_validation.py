import sys
import json
import pickle
import hashlib
from pathlib import Path

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from models.xcadta import XCADTA
from data_utils import smiles_to_graph, sequence_hash
from explain import generate_explanation


DEVICE = torch.device("cpu")

TARGET_SMILES = (
    "C1=CC(=CC=C1C2=C(NC(=N2)C3=CC=C(C=C3)F)"
    "C4=CC=NC=C4)F"
)

TARGET_PROTEIN_HASH = (
    "f400da208a3f5e988d1166d521230ca8db80a04de4798e3d1947845c3b0809df"
)

CHECKPOINT_PATH = ROOT / "checkpoints/cleaned_v1/xcadta_best.pt"
EMBEDDING_PATH = ROOT / "data/processed/esm_embeddings.pt"
DATASET_PATH = ROOT / "data/processed/cleaned_v1/test.pkl"

OUTPUT_DIR = ROOT / "results" / "final_figures"
OUTPUT_FILE = OUTPUT_DIR / "explainability_validation.json"


def make_json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, torch.Tensor):
        return make_json_safe(value.detach().cpu().tolist())

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())

    return value


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading candidate pair...")

    with open(DATASET_PATH, "rb") as f:
        test_data = pickle.load(f)

    matches = []

    for smiles, protein, affinity in test_data:
        protein_hash = hashlib.sha256(
            protein.encode()
        ).hexdigest()

        if (
            smiles == TARGET_SMILES
            and protein_hash == TARGET_PROTEIN_HASH
        ):
            matches.append(
                (smiles, protein, float(affinity))
            )

    if not matches:
        raise RuntimeError(
            "Target candidate was not found in cleaned_v1 test data."
        )

    smiles, sequence, true_affinity = matches[0]

    print("Candidate found.")
    print("Protein length:", len(sequence))
    print("True affinity:", true_affinity)

    print("\nLoading ESM-2 embeddings...")

    embeddings = torch.load(
        EMBEDDING_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    embedding_key = sequence_hash(sequence)

    if embedding_key not in embeddings:
        raise RuntimeError(
            "Candidate protein embedding not found in cache."
        )

    protein_emb = embeddings[embedding_key]

    if protein_emb.shape != (len(sequence), 1280):
        raise ValueError(
            f"Unexpected embedding shape: {tuple(protein_emb.shape)}"
        )

    protein_emb = protein_emb.float()

    print("Embedding shape:", tuple(protein_emb.shape))

    print("\nLoading trained XCADTA model...")

    model = XCADTA(
        drug_in_dim=65,
        esm_dim=1280,
        d_model=128,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint, strict=True)
    model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded successfully.")

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError("Invalid candidate SMILES.")

    graph = smiles_to_graph(smiles)

    print("\nStarting explainability evaluation...")
    print("SHAP permutations: 128")
    print("Integrated Gradients steps: 64")
    print("This may take some time on CPU.")

    explanation = generate_explanation(
        model=model,
        graph=graph,
        protein_emb=protein_emb,
        sequence=sequence,
        molecule=molecule,
        device=DEVICE,
        max_residues=20,
        permutations=128,
        ig_steps=64,
    )

    if explanation.get("status") != "complete":
        raise RuntimeError(
            f"Explanation did not complete: {explanation.get('status')}"
        )

    output = {
        "evaluation": {
            "project": "XCADTA",
            "candidate_protein_hash": embedding_key,
            "protein_length": len(sequence),
            "drug_smiles": smiles,
            "true_affinity": true_affinity,
            "checkpoint": str(
                CHECKPOINT_PATH.relative_to(ROOT)
            ),
            "device": str(DEVICE),
            "shap_permutations": 128,
            "integrated_gradients_steps": 64,
        },
        "explanation": explanation,
    }

    safe_output = make_json_safe(output)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            safe_output,
            f,
            indent=2,
            allow_nan=False,
        )

    print("\nEXPLAINABILITY EVALUATION COMPLETED")
    print("Output:", OUTPUT_FILE)

    print("\nACTUAL VALIDATION METRICS:")

    metrics = explanation.get("metrics", {})

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nJSON saved successfully.")


if __name__ == "__main__":
    main()
