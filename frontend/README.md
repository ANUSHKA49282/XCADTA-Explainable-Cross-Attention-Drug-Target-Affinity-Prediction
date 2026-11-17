# XCADTA Frontend

The XCADTA frontend is an interactive web dashboard built using React and Vite. It provides a user interface for drug–target affinity prediction, protein selection, model explainability and prediction history.

The frontend communicates with the XCADTA FastAPI backend to access prediction and explanation services.

## Features

- Drug–target affinity prediction interface
- SMILES input and molecular structure visualisation
- Cached protein library with search and selection
- Predicted affinity score display
- Residue-level attention visualisation
- Atom–residue cross-attention heatmap
- Explainability results and downloadable JSON reports
- Prediction history dashboard
- Model architecture and evaluation metrics

## Technology Stack

| Technology | Purpose |
|---|---|
| React | User interface |
| Vite | Development server and build tool |
| React Router | Client-side routing |
| Recharts | Data visualisation |
| Lucide React | Icons |
| RDKit.js | Molecular structure rendering |

## Project Structure

```text
frontend/
├── public/
├── src/
│   ├── assets/
│   ├── App.jsx
│   ├── App.css
│   ├── index.css
│   ├── main.jsx
│   └── MoleculeViewer.jsx
├── index.html
├── package.json
├── package-lock.json
├── vite.config.js
└── README.md
```

## Prerequisites

- Node.js
- npm
- XCADTA FastAPI backend

## Installation

Open a terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install the project dependencies:

```bash
npm install
```

## Running the Development Server

Start the Vite development server:

```bash
npm run dev
```

Open the local URL displayed in the terminal. By default, Vite commonly uses:

```text
http://localhost:5173
```

Keep the FastAPI backend running while using prediction and explainability features.

## Backend Connection

The frontend communicates with the XCADTA backend running locally at:

```text
http://127.0.0.1:8000
```

Before using the application, start the backend from the project root in a separate terminal:

```bash
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
```

The backend provides the prediction, protein library, explanation and history services.

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Production Build

Create an optimised production build:

```bash
npm run build
```

The generated production files will be available in:

```text
dist/
```

## Preview Production Build

To preview the production build locally:

```bash
npm run preview
```

## Linting

Run Oxlint to check the frontend source code:

```bash
npm run lint
```

## Available npm Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start the development server |
| `npm run build` | Generate the production build |
| `npm run preview` | Preview the production build |
| `npm run lint` | Run Oxlint checks |

## Important Notes

- The frontend requires the FastAPI backend for live predictions and explanation features.
- The trained model checkpoint and precomputed protein embeddings are managed by the backend.
- Install frontend dependencies using `npm install`.
- Do not remove `package-lock.json`; it helps maintain consistent dependency installation.
- The frontend does not independently train or evaluate the affinity prediction model.

---

**Project:** XCADTA  
**Frontend Framework:** React  
**Build Tool:** Vite  
**Backend:** FastAPI