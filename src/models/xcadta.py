
# XCADTA model with drug encoding, protein projection,
# bidirectional cross-attention and regression.
# Also includes a concatenation-based ablation model.

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from .drug_encoder import DrugEncoder
from .fusion import BidirectionalCrossAttentionFusion


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class XCADTA(nn.Module):
    def __init__(self, drug_in_dim: int, esm_dim: int = 1280, d_model: int = 128,
                 n_heads: int = 4, dropout: float = 0.2):
        super().__init__()

        self.drug_encoder = DrugEncoder(
            drug_in_dim,
            d_model,
            dropout=dropout
        )

        # Trainable projection of protein embeddings.
        self.protein_proj = nn.Linear(esm_dim, d_model)

        # Bidirectional cross-attention fusion.
        self.fusion = BidirectionalCrossAttentionFusion(d_model, n_heads)

        # Regression head receives four pooled feature vectors.
        self.head = RegressionHead(4 * d_model, dropout=dropout)

    def forward(self, graphs, protein_emb, protein_mask):
        # Generate atom-level drug representations.
        atom_h = self.drug_encoder(graphs.x, graphs.edge_index)

        # Convert atom representations into padded batches.
        drug_dense, drug_mask = to_dense_batch(atom_h, graphs.batch)

        # Project protein embeddings into the model dimension.
        protein_h = self.protein_proj(protein_emb)

        # Apply bidirectional cross-attention.
        pooled, attn_d2p = self.fusion(
            drug_dense,
            drug_mask,
            protein_h,
            protein_mask
        )

        # Predict binding affinity.
        pred = self.head(pooled)

        return pred, attn_d2p


class XCADTAConcatAblation(nn.Module):
    def __init__(self, drug_in_dim: int, esm_dim: int = 320, d_model: int = 128,
                 dropout: float = 0.2):
        super().__init__()

        self.drug_encoder = DrugEncoder(
            drug_in_dim,
            d_model,
            dropout=dropout
        )

        self.protein_proj = nn.Linear(esm_dim, d_model)

        # Regression head receives drug and protein mean-pooled features.
        self.head = RegressionHead(2 * d_model, dropout=dropout)

    @staticmethod
    def _masked_mean(x, mask):
        mask_f = mask.unsqueeze(-1).float()

        return (
            (x * mask_f).sum(dim=1)
            / mask_f.sum(dim=1).clamp(min=1e-6)
        )

    def forward(self, graphs, protein_emb, protein_mask):
        # Generate atom-level drug representations.
        atom_h = self.drug_encoder(graphs.x, graphs.edge_index)

        # Convert atom representations into padded batches.
        drug_dense, drug_mask = to_dense_batch(atom_h, graphs.batch)

        # Project protein embeddings.
        protein_h = self.protein_proj(protein_emb)

        # Apply masked mean pooling.
        drug_pooled = self._masked_mean(drug_dense, drug_mask)
        protein_pooled = self._masked_mean(protein_h, protein_mask)

        # Concatenate drug and protein representations.
        pooled = torch.cat(
            [drug_pooled, protein_pooled],
            dim=-1
        )

        # Predict binding affinity.
        pred = self.head(pooled)

        return pred, None
