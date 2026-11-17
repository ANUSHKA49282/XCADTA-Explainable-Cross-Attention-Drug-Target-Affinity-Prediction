# XCADTA

### Explainable Cross-Attention Drug–Target Affinity Prediction

XCADTA is a deep-learning project for predicting drug–target binding affinity from a small-molecule representation and a protein amino-acid sequence. It combines a graph-based drug encoder with pretrained ESM-2 protein embeddings and bidirectional cross-attention. A FastAPI backend exposes prediction and explanation endpoints, while a React + Vite frontend provides an interactive dashboard.

> **Research/demo note:** XCADTA produces computational affinity estimates and model-internal explanations. Attention or attribution scores should not be interpreted as experimentally confirmed binding sites or clinical evidence.

---

## Key Components

- **Drug Representation:** SMILES are converted into molecular graphs using RDKit.
- **Drug Encoder:** Graph Attention Network (GAT) and Graph Convolutional Network (GCN) layers learn molecular representations.
- **Protein Representation:** Precomputed ESM-2 embeddings provide 1280-dimensional residue representations.
- **Cross-Modal Interaction:** Bidirectional drug-to-protein and protein-to-drug cross-attention combines the two representations.
- **Affinity Prediction:** A fusion and regression head predicts a KIBA affinity score.
- **Explainability:** The application exposes model attention and residue-level explanation outputs. These describe model behaviour, not experimentally verified molecular contacts.
- **Application:** FastAPI serves the model; React and Vite provide the frontend; SQLite stores prediction history locally.

## Technology Stack

| Area | Technologies |
|---|---|
| Deep Learning | PyTorch, PyTorch Geometric |
| Molecular Processing | RDKit |
| Protein Embeddings | ESM-2, Transformers / fair-esm tooling |
| Explainability and Evaluation | SHAP, scikit-learn, SciPy |
| Data Processing | NumPy, Pandas, NetworkX |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite, Recharts, Lucide React |
| Local History | SQLite |

## Project Structure

The main project layout is organised as follows. Some data, model checkpoints and generated outputs are created locally and are not expected to be included in a source-code checkout.

```text
XCADTA/
├── data/
│   ├── processed/
│   │   ├── cleaned_v1/
│   │   └── esm_embeddings.pt
│   └── predictions.db
├── checkpoints/
│   └── cleaned_v1/
│       └── xcadta_best.pt
├── results/
│   ├── metrics_cleaned_test.json
│   ├── metrics_concat_ablation.json
│   └── final_figures/
├── scripts/
│   └── generate_figure_2_comparison.py
├── src/
│   ├── api.py
│   ├── train.py
│   ├── evaluate.py
│   ├── data_utils.py
│   └── models/
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
├── req.txt
└── README.md
```

The exact contents of `src/models/` and the data subdirectories may vary with the delivered project version. Keep the trained checkpoint and embedding cache at the paths shown above unless the paths in `src/api.py` are updated accordingly.

## Requirements

- Python 3.11 is recommended for the local backend environment.
- Node.js and npm for the frontend.
- The trained model checkpoint and precomputed protein embedding cache.
- The processed dataset files for model training and evaluation.

### Install Python Dependencies

Run this command from the project root:

```bash
pip install -r req.txt
```

For PyTorch and PyTorch Geometric, use installation instructions compatible with your operating system and CPU/CUDA setup if the default pip installation does not match your environment.

### Install Frontend Dependencies

```bash
cd frontend
npm install
```

## Run the Application Locally

Open two terminals from the project root.

### 1. Start the Backend

```bash
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
```

The API expects these files to exist:

```text
checkpoints/cleaned_v1/xcadta_best.pt
data/processed/esm_embeddings.pt
```

The backend creates `data/predictions.db` for local prediction history.

Once started, check the health endpoint:

```text
http://127.0.0.1:8000/health
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 2. Start the Frontend

In the second terminal:

```bash
cd frontend
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`).

Keep the backend running while using prediction and explanation features.

### Frontend Production Build

From the `frontend/` directory:

```bash
npm run build
npm run preview
```

Optional lint check:

```bash
npm run lint
```

## Model Training and Evaluation

The training and evaluation workflow used in the Colab notebook operates on the `cleaned_v1` processed dataset and the precomputed ESM-2 embedding file.

### Train XCADTA

```bash
python src/train.py \
  --data_dir data/processed/cleaned_v1 \
  --esm_embeddings data/processed/esm_embeddings.pt \
  --ckpt_dir checkpoints/cleaned_v1 \
  --model xcadta
```

### Train the Concatenation Ablation Model

```bash
python src/train.py \
  --data_dir data/processed/cleaned_v1 \
  --esm_embeddings data/processed/esm_embeddings.pt \
  --ckpt_dir checkpoints/organized/concat_ablation \
  --model concat_ablation
```

### Evaluate XCADTA

```bash
python src/evaluate.py \
  --data_dir data/processed/cleaned_v1 \
  --esm_embeddings data/processed/esm_embeddings.pt \
  --checkpoint checkpoints/cleaned_v1/xcadta_best.pt \
  --model xcadta \
  --out_json results/metrics_cleaned_test.json
```

### Evaluate the Concatenation Ablation

```bash
python src/evaluate.py \
  --data_dir data/processed/cleaned_v1 \
  --esm_embeddings data/processed/esm_embeddings.pt \
  --checkpoint checkpoints/organized/concat_ablation/concat_ablation_best.pt \
  --model concat_ablation \
  --out_json results/metrics_concat_ablation.json
```

These commands require the corresponding processed data, embeddings and training scripts. They are not required just to launch the already-trained local application.

## Evaluation Results

The final comparison generated by the project reports the following held-out evaluation metrics:

| Metric | XCADTA | Concat Ablation |
|---|---:|---:|
| Mean Squared Error (MSE) | 0.2478 | 0.3533 |
| Concordance Index (CI) | 0.8146 | 0.7761 |
| Modified R² (r2m) | 0.6086 | 0.4870 |

Lower MSE is better; higher CI and modified R² are better.

These values are tied to the evaluation outputs and split used for this project. Comparisons should only be interpreted as controlled ablations when both models use matching data splits and evaluation protocols.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Basic API information |
| `GET` | `/health` | Backend and model readiness |
| `GET` | `/proteins` | Available protein library |
| `GET` | `/drugs` | Available drug library |
| `POST` | `/predict` | Predict affinity for a drug–protein pair |
| `POST` | `/explain` | Generate explanation outputs |
| `GET` | `/history` | Retrieve locally stored prediction history |

See the interactive API documentation at `/docs` for request schemas and response details.

## Figures and Research Outputs

The final evaluation workflow produces five figures under `results/final_figures/`:

1. `Figure_1_XCADTA_Architecture.png`
2. `Figure_2_XCADTA_vs_Concat_Ablation.png`
3. `Figure_3_Attention_Heatmap.png`
4. `Figure_4_Explainability_Validation.png`
5. `Figure_5_Candidate_Residue_Attention.png`

The corresponding evaluation JSON files are written under `results/`. Keep the JSON outputs with the figures so reported values can be traced back to their evaluation results.

## Notes on Interpretation and Reproducibility

- The API uses a trained checkpoint and a precomputed ESM-2 embedding cache; installing packages alone does not recreate these artifacts.
- The protein sequence supplied to the prediction API must be represented in the local embedding cache used by the backend.
- The prediction history database is created locally and can be regenerated; it is not a trained model artifact.
- Attention, SHAP and other attribution outputs are model-level explanation signals. They do not establish experimentally validated binding residues.
- For a fair model comparison, retain the same dataset version, held-out split, preprocessing and evaluation settings for both XCADTA and the ablation model.

## Acknowledgements

The project uses the KIBA drug–target affinity dataset and pretrained ESM-2 protein representations. Please consult the original dataset and model documentation and follow their respective citation and usage requirements when redistributing or publishing results.