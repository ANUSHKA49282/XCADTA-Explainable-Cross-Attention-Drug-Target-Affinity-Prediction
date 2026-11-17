
# Drug encoder using GAT and GCN layers.
# Generates feature representations for each atom in a molecule.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class DrugEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, gat_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()

        # GAT layers with multiple attention heads.
        self.gat1 = GATConv(
            in_dim,
            d_model // gat_heads,
            heads=gat_heads,
            dropout=dropout
        )
        self.bn1 = nn.BatchNorm1d(d_model)

        self.gat2 = GATConv(
            d_model,
            d_model // gat_heads,
            heads=gat_heads,
            dropout=dropout
        )
        self.bn2 = nn.BatchNorm1d(d_model)

        # GCN layers for further graph feature extraction.
        self.gcn1 = GCNConv(d_model, d_model)
        self.bn3 = nn.BatchNorm1d(d_model)

        self.gcn2 = GCNConv(d_model, d_model)
        self.bn4 = nn.BatchNorm1d(d_model)

        self.dropout = dropout

    def forward(self, x, edge_index):
        # Apply the first GAT layer.
        h = self.gat1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Apply the second GAT layer.
        h = self.gat2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Apply the first GCN layer.
        h = self.gcn1(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Apply the second GCN layer.
        h = self.gcn2(h, edge_index)
        h = self.bn4(h)
        h = F.relu(h)

        # Return one embedding for each atom.
        return h
