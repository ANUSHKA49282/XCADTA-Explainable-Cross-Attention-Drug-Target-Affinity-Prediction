
import pickle
import hashlib
from pathlib import Path

# Project folder
ROOT = Path(__file__).resolve().parent.parent

TARGET_HASH = (
    "f400da208a3f5e988d1166d521230ca8db80a04de4798e3d1947845c3b0809df"
)

# Load cleaned test dataset
dataset_path = ROOT / "data/processed/cleaned_v1/test.pkl"

with open(dataset_path, "rb") as f:
    test = pickle.load(f)

# Find the candidate protein sequence
sequence_found = False

for smiles, sequence, affinity in test:

    protein_hash = hashlib.sha256(sequence.encode()).hexdigest()

    if protein_hash == TARGET_HASH:

        output = ROOT / "results/case_studies/candidate_protein_new.fasta"
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:
            f.write(">XCADTA_candidate_protein\n")

            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i + 60] + "\n")

        print("FASTA saved:", output)
        print("Sequence length:", len(sequence))

        sequence_found = True
        break

if not sequence_found:
    print("Protein sequence not found in test dataset.")
