
# Evaluation metrics for XCADTA and the concatenation ablation.
# Calculates MSE, Concordance Index and r2m.

import argparse
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import XCADTADataset, collate_fn
from models import XCADTA, XCADTAConcatAblation


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def concordance_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    chunk_size: int = 512
) -> float:
    # Compare pairs in chunks to reduce memory usage.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    n = len(y_true)

    if n != len(y_pred):
        raise ValueError("True and predicted values must have equal lengths")

    if n == 0:
        return float('nan')

    z = 0.0
    s = 0.0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        yt_chunk = y_true[start:end][:, None]
        yp_chunk = y_pred[start:end][:, None]

        gt_mask = yt_chunk > y_true[None, :]
        diff = yp_chunk - y_pred[None, :]

        z += gt_mask.sum()

        s += np.where(
            gt_mask & (diff > 0), 1.0, 0.0
        ).sum()

        s += np.where(
            gt_mask & (diff == 0), 0.5, 0.0
        ).sum()

    return float(s / z) if z > 0 else float('nan')


def r_squared_no_intercept(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # Calculate squared error for a regression line through the origin.
    denominator = np.sum(y_pred ** 2)

    if denominator == 0:
        return float('nan')

    k = np.sum(y_true * y_pred) / denominator
    y_pred_scaled = k * y_pred

    numerator = np.sum((y_true - y_pred_scaled) ** 2)
    total_variance = np.sum((y_true - np.mean(y_true)) ** 2)

    if total_variance == 0:
        return float('nan')

    return float(1 - numerator / total_variance)


def r2m(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Calculate the modified squared correlation coefficient.
    if len(y_true) < 2:
        return float('nan')

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float('nan')

    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r ** 2

    r0_2 = r_squared_no_intercept(y_true, y_pred)

    if not np.isfinite(r0_2):
        return float('nan')

    return float(r2 * (1 - np.sqrt(abs(r2 - r0_2))))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("True and predicted values must have equal lengths")

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty test set")

    if not np.all(np.isfinite(y_true)):
        raise ValueError("True values contain NaN or infinity")

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("Predictions contain NaN or infinity")

    return {
        'MSE': mse(y_true, y_pred),
        'CI': concordance_index(y_true, y_pred),
        'r2m': r2m(y_true, y_pred),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    for batch in loader:
        graphs = batch['graphs'].to(device)
        protein_emb = batch['protein_emb'].to(device)
        protein_mask = batch['protein_mask'].to(device)
        affinity = batch['affinity'].to(device)

        pred, _ = model(graphs, protein_emb, protein_mask)

        y_true.append(affinity.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    if not y_true:
        raise ValueError("The test data loader is empty")

    return np.concatenate(y_true), np.concatenate(y_pred)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--esm_embeddings', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)

    parser.add_argument(
        '--model',
        type=str,
        default='xcadta',
        choices=['xcadta', 'concat_ablation']
    )

    parser.add_argument('--drug_in_dim', type=int, default=65)
    parser.add_argument('--esm_dim', type=int, default=1280)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_json', type=str, default=None)

    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Device: {device} | Model: {args.model}")

    # Load precomputed protein embeddings.
    esm_embeddings = torch.load(
        args.esm_embeddings,
        map_location='cpu'
    )

    # Load the test dataset.
    test_path = os.path.join(args.data_dir, 'test.pkl')

    with open(test_path, 'rb') as f:
        test_pairs = pickle.load(f)

    test_ds = XCADTADataset(test_pairs, esm_embeddings)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize the selected model.
    if args.model == 'xcadta':
        model_class = XCADTA
    else:
        model_class = XCADTAConcatAblation

    model = model_class(
        drug_in_dim=args.drug_in_dim,
        esm_dim=args.esm_dim
    ).to(device)

    # Load the trained checkpoint.
    state_dict = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(state_dict)

    # Generate predictions and calculate metrics.
    y_true, y_pred = predict(model, test_loader, device)

    metrics = compute_all_metrics(y_true, y_pred)

    print(f"[{args.model}] Test metrics: {metrics}")

    # Save or update evaluation results.
    if args.out_json:
        output_dir = os.path.dirname(args.out_json)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(args.out_json):
            with open(args.out_json, 'r') as f:
                existing = json.load(f)
        else:
            existing = {}

        existing[args.model] = metrics

        with open(args.out_json, 'w') as f:
            json.dump(existing, f, indent=2)

        print(f"Saved evaluation results to {args.out_json}")


if __name__ == '__main__':
    main()
