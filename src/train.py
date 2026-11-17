
# Training loop for XCADTA and the concatenation ablation model.
# Uses AdamW, MSE loss, mixed precision and early stopping.

import argparse
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from data_utils import XCADTADataset, collate_fn
from models import XCADTA, XCADTAConcatAblation


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str, drug_in_dim: int, esm_dim: int):
    if name == "xcadta":
        return XCADTA(
            drug_in_dim=drug_in_dim,
            esm_dim=esm_dim
        )

    elif name == "concat_ablation":
        return XCADTAConcatAblation(
            drug_in_dim=drug_in_dim,
            esm_dim=esm_dim
        )

    raise ValueError(f"Unknown model name: {name}")


def run_epoch(model, loader, optimizer, scaler, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    n = 0

    criterion = torch.nn.MSELoss()

    with torch.set_grad_enabled(train):

        for batch in loader:

            graphs = batch["graphs"].to(device)
            protein_emb = batch["protein_emb"].to(device)
            protein_mask = batch["protein_mask"].to(device)
            affinity = batch["affinity"].to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):

                pred, _ = model(
                    graphs,
                    protein_emb,
                    protein_mask
                )

                loss = criterion(pred, affinity)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * affinity.size(0)
            n += affinity.size(0)

    return total_loss / max(1, n)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train.pkl and val.pkl"
    )

    parser.add_argument(
        "--esm_embeddings",
        type=str,
        required=True
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--model",
        type=str,
        default="xcadta",
        choices=["xcadta", "concat_ablation"]
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drug_in_dim", type=int, default=65)
    parser.add_argument("--esm_dim", type=int, default=1280)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    print(f"Device: {device} | Model: {args.model}")

    if use_amp:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("Mixed precision: enabled")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Load precomputed protein embeddings.
    esm_embeddings = torch.load(
        args.esm_embeddings,
        map_location="cpu"
    )

    # Load training and validation datasets.
    with open(
        os.path.join(args.data_dir, "train.pkl"), "rb"
    ) as f:
        train_pairs = pickle.load(f)

    with open(
        os.path.join(args.data_dir, "val.pkl"), "rb"
    ) as f:
        val_pairs = pickle.load(f)

    train_ds = XCADTADataset(
        train_pairs,
        esm_embeddings
    )

    val_ds = XCADTADataset(
        val_pairs,
        esm_embeddings
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    # Initialize model.
    model = build_model(
        args.model,
        args.drug_in_dim,
        args.esm_dim
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Modern AMP GradScaler API.
    scaler = GradScaler(
        "cuda",
        enabled=use_amp
    )

    best_val = float("inf")
    patience_counter = 0
    start_epoch = 0

    # Checkpoint paths.
    best_path = os.path.join(
        args.ckpt_dir,
        f"{args.model}_best.pt"
    )

    resume_path = os.path.join(
        args.ckpt_dir,
        f"{args.model}_last.pt"
    )

    # Resume from the latest checkpoint if available.
    if os.path.exists(resume_path):

        print(f"Resuming from {resume_path}")

        ckpt = torch.load(
            resume_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            ckpt["model_state"]
        )

        optimizer.load_state_dict(
            ckpt["optimizer_state"]
        )

        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        patience_counter = ckpt["patience_counter"]

        if "scaler_state" in ckpt:
            scaler.load_state_dict(
                ckpt["scaler_state"]
            )

        print(f"Resuming from epoch {start_epoch}")

    # Training loop.
    for epoch in range(start_epoch, args.max_epochs):

        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            train=True
        )

        val_loss = run_epoch(
            model,
            val_loader,
            optimizer,
            scaler,
            device,
            train=False
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train MSE: {train_loss:.6f} | "
            f"val MSE: {val_loss:.6f}"
        )

        # Save best model.
        improved = val_loss < best_val

        if improved:

            best_val = val_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                best_path
            )

            print("Best checkpoint updated.")

        else:
            patience_counter += 1

        # Save latest training state.
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_val": best_val,
                "patience_counter": patience_counter,
            },
            resume_path
        )

        # Early stopping.
        if patience_counter >= args.patience:

            print(
                f"Early stopping at epoch {epoch} "
                f"(patience={args.patience})"
            )

            break

    print("\nTraining completed.")

    print(f"Best validation MSE: {best_val:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Latest checkpoint: {resume_path}")


if __name__ == "__main__":
    main()
