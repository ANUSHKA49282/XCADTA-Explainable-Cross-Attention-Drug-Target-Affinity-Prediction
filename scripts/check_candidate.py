
import pickle
import hashlib
import csv
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

# Project folder
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data_utils import smiles_to_graph, sequence_hash
from src.models.xcadta import XCADTA

DEVICE = torch.device("cpu")

TARGET_SMILES = (
    "C1=CC(=CC=C1C2=C(NC(=N2)C3=CC=C(C=C3)F)"
    "C4=CC=NC=C4)F"
)

TARGET_PROTEIN_HASH = (
    "f400da208a3f5e988d1166d521230ca8db80a04de4798e3d1947845c3b0809df"
)

# 1. Load test dataset
dataset_path = ROOT / "data/processed/cleaned_v1/test.pkl"

with open(dataset_path, "rb") as f:
    test = pickle.load(f)

# Find the required drug-protein pair
matches = []

for smiles, protein, affinity in test:
    protein_hash = hashlib.sha256(protein.encode()).hexdigest()

    if smiles == TARGET_SMILES and protein_hash == TARGET_PROTEIN_HASH:
        matches.append((smiles, protein, float(affinity)))

if len(matches) == 0:
    print("Drug-protein pair not found.")
    sys.exit()

smiles, sequence, y_true = matches[0]

# 2. Load protein embeddings
embedding_path = ROOT / "data/processed/esm_embeddings.pt"

embeddings = torch.load(
    embedding_path,
    map_location=DEVICE,
    weights_only=True
)

embedding_key = sequence_hash(sequence)

if embedding_key not in embeddings:
    print("Protein embedding not found.")
    sys.exit()

embedding = embeddings[embedding_key]

if embedding.shape[0] != len(sequence):
    print("Protein and embedding lengths do not match.")
    sys.exit()

protein_emb = embedding.unsqueeze(0)

protein_mask = torch.ones(
    (1, len(sequence)),
    dtype=torch.bool
)

# 3. Load trained model
checkpoint_path = ROOT / "checkpoints/cleaned_v1/xcadta_best.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True
)

model = XCADTA(
    drug_in_dim=65,
    esm_dim=1280,
    d_model=128
)

model.load_state_dict(checkpoint, strict=True)
model.eval()

# 4. Convert drug SMILES into a graph
graph = smiles_to_graph(smiles)
graph_batch = Batch.from_data_list([graph])

valid_atom_count = graph.x.shape[0]

# 5. Predict affinity and get attention values
with torch.no_grad():
    prediction, attention = model(
        graph_batch,
        protein_emb,
        protein_mask
    )

y_pred = prediction.item()

if attention is None:
    print("Attention values not available.")
    sys.exit()

if attention.shape != (1, valid_atom_count, len(sequence)):
    print("Unexpected attention shape:", attention.shape)
    sys.exit()

# 6. Calculate residue importance
attention_matrix = attention[0].cpu().numpy()

residue_importance = attention_matrix.mean(axis=0)

# 7. Rank protein residues
ranked_indices = residue_importance.argsort()[::-1]

output_dir = ROOT / "results" / "case_studies"
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "candidate_residue_attention_new.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "sequence_position",
        "amino_acid",
        "attention_importance",
        "attention_rank"
    ])

    for rank, index in enumerate(ranked_indices):
        writer.writerow([
            int(index) + 1,
            sequence[index],
            float(residue_importance[index]),
            rank + 1
        ])

# 8. Display results
print("\n--XCADTA CASE STUDY--")

print("Pair found: Yes")
print("Protein length:", len(sequence))
print("Valid drug atoms:", valid_atom_count)
print("True affinity:", y_true)
print("Predicted affinity:", round(y_pred, 6))
print("Absolute error:", round(abs(y_pred - y_true), 6))
print("Attention shape:", tuple(attention.shape))

print("\nTop 10 important residues:")

for index in ranked_indices[:10]:
    print(
        "Position:", index + 1,
        "| Amino acid:", sequence[index],
        "| Importance:", round(float(residue_importance[index]), 8)
    )

print("\nCSV saved:", csv_path)
print("Checkpoint:", checkpoint_path)
