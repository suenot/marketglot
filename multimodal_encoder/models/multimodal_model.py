from __future__ import annotations
import torch
import torch.nn as nn


class MultimodalEncoder(nn.Module):
    """End-to-end multimodal model: candle encoder + indicator encoder -> fusion transformer."""

    def __init__(
        self,
        # Candle params
        delta_vocab_size: int = 122,
        bucket_vocab_size: int = 10,
        delta_emb_dim: int = 64,
        bucket_emb_dim: int = 16,
        candle_proj_dim: int = 128,
        # Indicator params
        ind_vocab_sizes: list[int] | None = None,
        ind_emb_dim: int = 16,
        ind_proj_dim: int = 128,
        # Fusion params
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 3,
        seq_len: int = 128,
    ) -> None:
        super().__init__()
        if ind_vocab_sizes is None:
            ind_vocab_sizes = [7, 9, 7, 8, 7, 7]

        # Candle encoder
        self.delta_emb = nn.Embedding(delta_vocab_size, delta_emb_dim)
        self.vol_emb = nn.Embedding(bucket_vocab_size, bucket_emb_dim)
        self.vb_emb = nn.Embedding(bucket_vocab_size, bucket_emb_dim)
        candle_concat = delta_emb_dim + bucket_emb_dim * 2
        self.candle_proj = nn.Linear(candle_concat, candle_proj_dim)

        # Indicator encoder
        self.ind_embeddings = nn.ModuleList([
            nn.Embedding(vs, ind_emb_dim) for vs in ind_vocab_sizes
        ])
        ind_concat = ind_emb_dim * len(ind_vocab_sizes)
        self.ind_proj = nn.Linear(ind_concat, ind_proj_dim)

        # Fusion
        fusion_dim = candle_proj_dim + ind_proj_dim
        self.fusion_proj = nn.Linear(fusion_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_emb = nn.Embedding(seq_len + 1, hidden_dim)  # +1 for CLS

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        delta_ids: torch.Tensor,
        vol_ids: torch.Tensor,
        vb_ids: torch.Tensor,
        ind_inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        B = delta_ids.shape[0]

        # Candle stream
        d = self.delta_emb(delta_ids)
        v = self.vol_emb(vol_ids)
        vb = self.vb_emb(vb_ids)
        candle = self.candle_proj(torch.cat([d, v, vb], dim=-1))  # (B, T, candle_proj_dim)

        # Indicator stream
        ind_embs = [emb(tok) for emb, tok in zip(self.ind_embeddings, ind_inputs)]
        indicator = self.ind_proj(torch.cat(ind_embs, dim=-1))  # (B, T, ind_proj_dim)

        # Fuse per-position
        fused = self.fusion_proj(torch.cat([candle, indicator], dim=-1))  # (B, T, hidden_dim)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, fused], dim=1)  # (B, T+1, hidden_dim)

        # Positional embeddings
        T = x.shape[1]
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # CLS pooling
        return self.head(x[:, 0])
