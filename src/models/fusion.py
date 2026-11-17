import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-head cross-attention
class CrossAttention(nn.Module):

    def __init__(self, d_model=128, n_heads=4):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q_src, kv_src, kv_mask=None):

        B, Lq, _ = q_src.shape
        Lkv = kv_src.shape[1]

        # Query, key and value
        Q = self.W_q(q_src).view(
            B, Lq, self.n_heads, self.d_head
        ).transpose(1, 2)

        K = self.W_k(kv_src).view(
            B, Lkv, self.n_heads, self.d_head
        ).transpose(1, 2)

        V = self.W_v(kv_src).view(
            B, Lkv, self.n_heads, self.d_head
        ).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_head)

        if kv_mask is not None:
            mask = kv_mask[:, None, None, :]
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(
            B, Lq, self.d_model
        )

        out = self.W_o(out)

        # Average attention across heads
        attn_avg = attn.mean(dim=1)

        return out, attn_avg


# Bidirectional drug-protein attention
class BidirectionalCrossAttentionFusion(nn.Module):

    def __init__(self, d_model=128, n_heads=4):
        super().__init__()

        self.d2p = CrossAttention(d_model, n_heads)
        self.p2d = CrossAttention(d_model, n_heads)

        self.last_attn_d2p = None
        self.last_attn_p2d = None

    # Mean and max pooling
    @staticmethod
    def _masked_mean_max(x, mask):

        mask_f = mask.unsqueeze(-1).float()

        summed = (x * mask_f).sum(dim=1)
        count = mask_f.sum(dim=1).clamp(min=1e-6)

        mean = summed / count

        x_masked = x.masked_fill(
            ~mask.unsqueeze(-1), float('-inf')
        )

        maxed = x_masked.max(dim=1).values
        maxed = torch.nan_to_num(maxed, neginf=0.0)

        return mean, maxed

    def forward(self, drug_atoms, drug_mask, protein_res, protein_mask):

        # Drug to protein attention
        updated_drug, attn_d2p = self.d2p(
            drug_atoms, protein_res, kv_mask=protein_mask
        )

        # Protein to drug attention
        updated_protein, attn_p2d = self.p2d(
            protein_res, drug_atoms, kv_mask=drug_mask
        )

        # Store attention values for explainability
        self.last_attn_d2p = attn_d2p.detach()
        self.last_attn_p2d = attn_p2d.detach()

        # Pool drug and protein features
        drug_mean, drug_max = self._masked_mean_max(
            updated_drug, drug_mask
        )

        prot_mean, prot_max = self._masked_mean_max(
            updated_protein, protein_mask
        )

        # Combine all features
        pooled = torch.cat(
            [drug_mean, drug_max, prot_mean, prot_max],
            dim=-1
        )

        return pooled, attn_d2p
