import json
import pickle
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from torch_geometric.data import Batch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from models.xcadta import XCADTA
from data_utils import smiles_to_graph, sequence_hash
from explain import generate_explanation


# --------------------------------------------------
# PATHS AND MODEL CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "cleaned_v1" / "xcadta_best.pt"
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "esm_embeddings.pt"
SEQUENCES_PATH = ROOT / "data" / "processed" / "unique_sequences.pkl"
LIGANDS_PATH = ROOT / "data" / "raw" / "kiba" / "ligands_can.txt"
DATABASE_PATH = ROOT / "data" / "predictions.db"

DEVICE = torch.device("cpu")
ESM_DIM = 1280
D_MODEL = 128
N_HEADS = 4
DROPOUT = 0.2

ALLOWED_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")

model = None
protein_embeddings = None
protein_library = []
drug_library = []


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def initialize_database():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                smiles TEXT NOT NULL,
                protein_sequence TEXT NOT NULL,
                affinity REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_prediction(smiles, sequence, affinity):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO predictions (smiles, protein_sequence, affinity)
            VALUES (?, ?, ?)
            """,
            (smiles, sequence, affinity),
        )


# --------------------------------------------------
# PROTEIN LIBRARY EXTRACTION
# --------------------------------------------------

def collect_sequences(obj, found):
    """Recursively find sequences whose exact embeddings exist in the cache."""

    if isinstance(obj, str):
        sequence = obj.strip().upper()

        if (
            len(sequence) >= 20
            and all(letter in ALLOWED_AMINO_ACIDS for letter in sequence)
        ):
            key = sequence_hash(sequence)

            if key in protein_embeddings:
                found[key] = sequence

    elif isinstance(obj, dict):
        for key, value in obj.items():
            collect_sequences(key, found)
            collect_sequences(value, found)

    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            collect_sequences(item, found)

    elif hasattr(obj, "to_dict"):
        collect_sequences(obj.to_dict(), found)

    elif hasattr(obj, "tolist"):
        collect_sequences(obj.tolist(), found)


def load_protein_library():
    found = {}

    if not SEQUENCES_PATH.exists():
        print(f"Protein sequence source not found: {SEQUENCES_PATH}")
        return []

    try:
        with open(SEQUENCES_PATH, "rb") as file:
            source_data = pickle.load(file)

        collect_sequences(source_data, found)

    except Exception as error:
        print(f"Could not read protein sequence source: {error}")

    proteins = [
        {
            "id": key,
            "sequence": sequence,
            "length": len(sequence),
            "label": f"Protein · {len(sequence)} residues · {key[:8]}",
        }
        for key, sequence in found.items()
    ]

    proteins.sort(key=lambda item: (item["length"], item["id"]))
    return proteins


def load_drug_library():
    """Load real KIBA compounds from ligands_can.txt (drug ID -> SMILES)."""
    if not LIGANDS_PATH.exists():
        print(f"KIBA ligand file not found: {LIGANDS_PATH}")
        return []

    try:
        with open(LIGANDS_PATH, "r", encoding="utf-8") as file:
            raw_ligands = json.load(file)
    except Exception as error:
        print(f"Could not read KIBA ligand library: {error}")
        return []

    drugs = []
    if isinstance(raw_ligands, dict):
        for drug_id, smiles in raw_ligands.items():
            if not isinstance(smiles, str) or not smiles.strip():
                continue
            molecule = Chem.MolFromSmiles(smiles.strip())
            if molecule is None:
                continue
            drugs.append({
                "id": str(drug_id),
                "label": str(drug_id),
                "smiles": smiles.strip(),
                "atoms": int(molecule.GetNumAtoms()),
            })

    drugs.sort(key=lambda item: item["id"].lower())
    return drugs


# --------------------------------------------------
# LOAD TRAINED MODEL AND EMBEDDINGS
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, protein_embeddings, protein_library, drug_library

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    if not EMBEDDINGS_PATH.exists():
        raise RuntimeError(f"Embedding cache not found: {EMBEDDINGS_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    protein_embeddings = torch.load(
        EMBEDDINGS_PATH,
        map_location="cpu",
        weights_only=False,
    )

    gat_weight = checkpoint.get("drug_encoder.gat1.lin.weight")

    if gat_weight is None:
        raise RuntimeError(
            "Could not identify the first GAT layer in the checkpoint."
        )

    drug_input_dim = gat_weight.shape[1]

    model = XCADTA(
        drug_in_dim=drug_input_dim,
        esm_dim=ESM_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)

    model.load_state_dict(checkpoint)
    model.eval()

    initialize_database()
    protein_library = load_protein_library()
    drug_library = load_drug_library()

    print("XCADTA model loaded successfully.")
    print(f"Device: {DEVICE}")
    print(f"Protein embedding dimension: {ESM_DIM}")
    print(f"Cached protein embeddings: {len(protein_embeddings)}")
    print(f"Protein sequences available in selector: {len(protein_library)}")
    print(f"KIBA drug compounds available in selector: {len(drug_library)}")

    yield

    model = None
    protein_embeddings = None
    protein_library = []
    drug_library = []


app = FastAPI(
    title="XCADTA API",
    description="Explainable Cross-Attention Drug–Target Affinity Prediction",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# REQUEST SCHEMA
# --------------------------------------------------

class PredictionRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=5000)
    protein_sequence: str = Field(min_length=1, max_length=10000)


class ExplanationRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=5000)
    protein_sequence: str = Field(min_length=1, max_length=10000)
    max_residues: int = Field(default=20, ge=5, le=32)
    permutations: int = Field(default=128, ge=16, le=256)
    ig_steps: int = Field(default=64, ge=32, le=256)


# --------------------------------------------------
# HEALTH AND PROTEIN LIBRARY
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "XCADTA API is running",
        "model": "XCADTA",
        "device": str(DEVICE),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "not_ready",
        "model_loaded": model is not None,
        "cached_proteins": len(protein_embeddings) if protein_embeddings is not None else 0,
        "selectable_proteins": len(protein_library),
    }


@app.get("/proteins")
def get_proteins():
    return {
        "total": len(protein_library),
        "proteins": protein_library,
    }


@app.get("/drugs")
def get_drugs():
    return {
        "total": len(drug_library),
        "drugs": drug_library,
    }


# --------------------------------------------------
# DRUG–TARGET AFFINITY PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or protein_embeddings is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    smiles = request.smiles.strip()
    sequence = "".join(request.protein_sequence.split()).upper()

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid SMILES. Please enter a valid molecular structure.",
        )

    if not sequence or any(letter not in ALLOWED_AMINO_ACIDS for letter in sequence):
        raise HTTPException(
            status_code=400,
            detail="Invalid protein sequence. Check the amino acid letters.",
        )

    protein_key = sequence_hash(sequence)

    if protein_key not in protein_embeddings:
        raise HTTPException(
            status_code=404,
            detail=(
                "This protein sequence is not present in the local ESM-2 cache. "
                "Choose a sequence from the protein library."
            ),
        )

    protein_emb = protein_embeddings[protein_key]

    if protein_emb.ndim != 2 or protein_emb.shape[-1] != ESM_DIM:
        raise HTTPException(
            status_code=500,
            detail="Cached protein embedding has an incompatible shape.",
        )

    try:
        drug_graph = smiles_to_graph(smiles)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not construct molecular graph: {error}",
        )

    graphs = Batch.from_data_list([drug_graph]).to(DEVICE)
    protein_tensor = protein_emb.unsqueeze(0).to(DEVICE)

    protein_mask = torch.ones(
        (1, protein_tensor.shape[1]),
        dtype=torch.bool,
        device=DEVICE,
    )

    try:
        with torch.inference_mode():
            prediction, attention = model(
                graphs,
                protein_tensor,
                protein_mask,
            )

        affinity = float(prediction.squeeze().cpu().item())

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {error}",
        )

    attention_data = []
    heatmap_data = None

    if attention is not None:
        # Expected shape: (batch, drug_atoms, protein_residues)
        attention_matrix = attention.squeeze(0).detach().cpu()

        if attention_matrix.ndim == 2:
            residue_attention = attention_matrix.mean(dim=0)
            top_count = min(20, len(sequence))

            top_indices = torch.topk(
                residue_attention,
                k=top_count,
            ).indices.tolist()

            attention_data = [
                {
                    "position": index + 1,
                    "amino_acid": sequence[index],
                    "attention": float(residue_attention[index]),
                }
                for index in top_indices
            ]

            attention_data.sort(
                key=lambda item: item["attention"],
                reverse=True,
            )

            sorted_indices = [
                item["position"] - 1 for item in attention_data
            ]

            heatmap_data = {
                "residues": [
                    {
                        "position": item["position"],
                        "amino_acid": item["amino_acid"],
                    }
                    for item in attention_data
                ],
                "atoms": [
                    [
                        float(attention_matrix[atom_index, residue_index])
                        for residue_index in sorted_indices
                    ]
                    for atom_index in range(attention_matrix.shape[0])
                ],
            }

    save_prediction(smiles, sequence, affinity)

    return {
        "success": True,
        "prediction": {
            "affinity": affinity,
            "unit": "KIBA score",
        },
        "drug": {
            "smiles": smiles,
            "atoms": int(drug_graph.x.shape[0]),
        },
        "protein": {
            "sequence_length": len(sequence),
            "embedding_dimension": ESM_DIM,
            "sequence": sequence,
        },
        "explainability": {
            "method": "Drug-to-protein cross-attention",
            "top_residues": attention_data,
            "atom_residue_heatmap": heatmap_data,
            "note": (
                "Attention weights describe model attention, "
                "not experimentally confirmed binding contacts."
            ),
        },
    }


# --------------------------------------------------
# FULL MODEL-GROUNDED EXPLANATION
# --------------------------------------------------

@app.post("/explain")
def explain_prediction(request: ExplanationRequest):
    """Compute atom Integrated Gradients and residue permutation-SHAP diagnostics."""
    if model is None or protein_embeddings is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    smiles = request.smiles.strip()
    sequence = "".join(request.protein_sequence.split()).upper()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES. Please enter a valid molecular structure.")
    if not sequence or any(letter not in ALLOWED_AMINO_ACIDS for letter in sequence):
        raise HTTPException(status_code=400, detail="Invalid protein sequence. Check the amino acid letters.")

    protein_key = sequence_hash(sequence)
    if protein_key not in protein_embeddings:
        raise HTTPException(
            status_code=404,
            detail="This protein sequence is not present in the local ESM-2 cache. Choose a sequence from the protein library.",
        )

    protein_emb = protein_embeddings[protein_key]
    if protein_emb.ndim != 2 or protein_emb.shape[-1] != ESM_DIM:
        raise HTTPException(status_code=500, detail="Cached protein embedding has an incompatible shape.")

    try:
        drug_graph = smiles_to_graph(smiles)
        explanation = generate_explanation(
            model=model,
            graph=drug_graph,
            protein_emb=protein_emb,
            sequence=sequence,
            molecule=molecule,
            device=DEVICE,
            max_residues=request.max_residues,
            permutations=request.permutations,
            ig_steps=request.ig_steps,
        )
        return {
            "success": True,
            "drug": {"smiles": smiles, "atoms": int(drug_graph.x.shape[0])},
            "protein": {"sequence_length": len(sequence)},
            "explanation": explanation,
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Explanation computation failed: {error}")



# --------------------------------------------------
# SIMILAR COMPOUND RECOMMENDATIONS
# --------------------------------------------------

class RecommendationRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=5000)
    protein_sequence: str = Field(min_length=1, max_length=10000)
    limit: int = Field(default=5, ge=1, le=10)


@app.post("/recommendations")
def recommend_similar_compounds(request: RecommendationRequest):
    """Find real KIBA ligands by Morgan/Tanimoto similarity and score them with XCADTA."""
    if model is None or protein_embeddings is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    query_smiles = request.smiles.strip()
    sequence = "".join(request.protein_sequence.split()).upper()
    query_molecule = Chem.MolFromSmiles(query_smiles)
    if query_molecule is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES for similarity search.")
    if not sequence or any(letter not in ALLOWED_AMINO_ACIDS for letter in sequence):
        raise HTTPException(status_code=400, detail="Invalid protein sequence.")
    if not drug_library:
        raise HTTPException(status_code=503, detail="The KIBA compound library is unavailable.")

    protein_key = sequence_hash(sequence)
    if protein_key not in protein_embeddings:
        raise HTTPException(
            status_code=404,
            detail="This protein sequence is not present in the local ESM-2 cache. Choose a sequence from the protein library.",
        )

    protein_emb = protein_embeddings[protein_key]
    if protein_emb.ndim != 2 or protein_emb.shape[-1] != ESM_DIM or protein_emb.shape[0] != len(sequence):
        raise HTTPException(status_code=500, detail="Cached protein embedding has an incompatible shape.")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query_fingerprint = generator.GetFingerprint(query_molecule)
    query_canonical = Chem.MolToSmiles(query_molecule, canonical=True)

    ranked = []
    seen_canonical = {query_canonical}
    for drug in drug_library:
        candidate_molecule = Chem.MolFromSmiles(drug["smiles"])
        if candidate_molecule is None:
            continue
        canonical = Chem.MolToSmiles(candidate_molecule, canonical=True)
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        candidate_fingerprint = generator.GetFingerprint(candidate_molecule)
        similarity = float(
            DataStructs.TanimotoSimilarity(query_fingerprint, candidate_fingerprint)
        )
        ranked.append((similarity, drug, candidate_molecule))

    ranked.sort(key=lambda item: item[0], reverse=True)
    shortlist = ranked[: min(20, len(ranked))]

    scored = []
    for similarity, drug, molecule in shortlist:
        try:
            graph = smiles_to_graph(drug["smiles"])
            graph_batch = Batch.from_data_list([graph]).to(DEVICE)
            protein_batch = protein_emb.unsqueeze(0).to(DEVICE)
            mask = torch.ones(
                (1, protein_emb.shape[0]), dtype=torch.bool, device=DEVICE
            )
            with torch.inference_mode():
                prediction, _ = model(graph_batch, protein_batch, mask)
            affinity = float(prediction.reshape(-1)[0].cpu().item())
            scored.append({
                "id": drug["id"],
                "smiles": drug["smiles"],
                "atoms": int(molecule.GetNumAtoms()),
                "tanimoto_similarity": similarity,
                "predicted_affinity": affinity,
            })
        except Exception:
            continue

    # KIBA higher scores indicate stronger predicted interaction; show strongest first.
    scored.sort(key=lambda item: item["predicted_affinity"], reverse=True)
    return {
        "success": True,
        "method": "Morgan fingerprints (radius 2, 2048 bits) with Tanimoto similarity",
        "ranking": "XCADTA-predicted KIBA score, descending",
        "recommendations": scored[:request.limit],
        "note": (
            "These are computational shortlist candidates from the local KIBA ligand library. "
            "Similarity and predicted scores are not experimental evidence of binding or safety."
        ),
    }


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

@app.get("/history")
def get_history():
    if not DATABASE_PATH.exists():
        return {"predictions": []}

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT id, smiles, protein_sequence, affinity, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()

    return {"predictions": [dict(row) for row in rows]}
