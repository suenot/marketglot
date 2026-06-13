from __future__ import annotations
import torch
import torch.nn as nn


class IndicatorModel(nn.Module):
    """Model B: encodes indicator token sequences into 3-class predictions."""

    def __init__(
        self,
        vocab_sizes: list[int],
        emb_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 3,
        seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, emb_dim) for vs in vocab_sizes
        ])
        n_indicators = len(vocab_sizes)
        concat_dim = emb_dim * n_indicators
        self.proj = nn.Linear(concat_dim, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len + 1, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
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

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        B = inputs[0].shape[0]
        embs = [emb(tok) for emb, tok in zip(self.embeddings, inputs)]
        x = torch.cat(embs, dim=-1)
        x = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        T = x.shape[1]
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(positions)
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x[:, 0])
