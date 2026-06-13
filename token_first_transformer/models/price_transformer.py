from __future__ import annotations

import torch
import torch.nn as nn


class PriceTransformer(nn.Module):
    def __init__(
        self,
        delta_vocab_size: int = 122,
        bucket_vocab_size: int = 10,
        delta_emb_dim: int = 64,
        bucket_emb_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 3,
        seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.delta_emb = nn.Embedding(delta_vocab_size, delta_emb_dim)
        self.vol_emb = nn.Embedding(bucket_vocab_size, bucket_emb_dim)
        self.vb_emb = nn.Embedding(bucket_vocab_size, bucket_emb_dim)

        concat_dim = delta_emb_dim + bucket_emb_dim * 2
        self.proj = nn.Linear(concat_dim, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self, delta_ids: torch.Tensor, vol_ids: torch.Tensor, vb_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, T = delta_ids.shape
        d = self.delta_emb(delta_ids)
        v = self.vol_emb(vol_ids)
        vb = self.vb_emb(vb_ids)
        x = torch.cat([d, v, vb], dim=-1)
        x = self.proj(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(positions)
        x = self.transformer(x)
        x = self.layer_norm(x)
        cls_out = x[:, 0, :]
        return self.head(cls_out)
