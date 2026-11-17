import pickle
import hashlib
from pathlib import Path
from rdkit import Chem

# Project folder path
ROOT = Path(__file__).resolve().parent.parent

TARGET_HASH = "f400da208a3f5e988d1166d521230ca8db80a04de4798e3d1947845c3b0809df"

TARGET_SMILES = (
    "C1=CC(=CC=C1C2=C(NC(=N2)C3=CC=C(C=C3)F)"
    "C4=CC=NC=C4)F"
)

# Load test dataset
dataset_path = ROOT / "data/processed/cleaned_v1/test.pkl"

with open(dataset_path, "rb") as f:
    test = pickle.load(f)

# Find the candidate drug-protein pair
matches = []

for smiles, protein, affinity in test:
    protein_hash = hashlib.sha256(protein.encode()).hexdigest()

    if smiles == TARGET_SMILES and protein_hash == TARGET_HASH:
        matches.append((smiles, protein, float(affinity)))

if len(matches) == 0:
    print("Candidate not found in test dataset.")
    exit()

smiles, protein, affinity = matches[0]

# Check drug structure
mol = Chem.MolFromSmiles(smiles)

if mol is None:
    print("Invalid SMILES.")
    exit()

canonical_smiles = Chem.MolToSmiles(mol)
inchikey = Chem.MolToInchiKey(mol)

# Display results
print("\n-- CANDIDATE IDENTITY AUDIT--")

print("\nDrug Information")
print("Original SMILES:", smiles)
print("Canonical SMILES:", canonical_smiles)
print("InChIKey:", inchikey)
print("Number of atoms:", mol.GetNumAtoms())
print("Number of bonds:", mol.GetNumBonds())

print("\nProtein Information")
print("Protein length:", len(protein))
print("Protein SHA256:", hashlib.sha256(protein.encode()).hexdigest())
print("First 40 residues:", protein[:40])
print("Last 40 residues:", protein[-40:])

print("\nDataset Information")
print("Recorded affinity:", affinity)
