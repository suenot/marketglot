from __future__ import annotations

import torch
import torch.nn as nn

from models.moe_layer import MoELayer


class MoETransformerBlock(nn.Module):
    """Single transformer block: LN -> MHA -> residual -> LN -> MoE -> residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = MoELayer(dim, hidden_dim, num_experts, top_k)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns output and aux_loss."""
        # Self-attention with residual
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.dropout(attn_out)

        # MoE FFN with residual
        residual = x
        x_norm = self.norm2(x)
        moe_out, aux_loss = self.moe(x_norm)
        x = residual + self.dropout(moe_out)

        return x, aux_loss


class MoETradingModel(nn.Module):
    """Mixture of Experts transformer for 3-class trading prediction."""

    def __init__(
        self,
        seq_len: int = 128,
        num_experts: int = 8,
        top_k: int = 2,
        num_layers: int = 4,
        num_heads: int = 8,
        dim: int = 256,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim

        # Candle encoder
        # delta_emb(122, 64) + vol_emb(10, 16) + vb_emb(10, 16) = 96
        self.delta_emb = nn.Embedding(122, 64)
        self.vol_emb = nn.Embedding(10, 16)
        self.vb_emb = nn.Embedding(10, 16)
        self.candle_proj = nn.Linear(96, 128)

        # Indicator encoder: 6 embeddings, each vocab_size -> 16, concat = 96
        # vocab_sizes: RSI=7, MACD=9, BB=7, ATR=8, VR=7, PVS=7
        self.indicator_embs = nn.ModuleList([
            nn.Embedding(7, 16),   # RSI
            nn.Embedding(9, 16),   # MACD
            nn.Embedding(7, 16),   # BB
            nn.Embedding(8, 16),   # ATR
            nn.Embedding(7, 16),   # VR
            nn.Embedding(7, 16),   # PVS
        ])
        self.indicator_proj = nn.Linear(96, 128)

        # Fusion: concat candle + indicator -> (batch, seq_len, 256)
        # No separate projection needed since dim=256

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Positional embedding for seq_len + 1 (CLS)
        self.pos_emb = nn.Embedding(seq_len + 1, dim)

        # MoE Transformer layers
        self.layers = nn.ModuleList([
            MoETransformerBlock(dim, num_heads, hidden_dim, num_experts, top_k, dropout)
            for _ in range(num_layers)
        ])

        # CLS pooling + MLP head
        self.head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        delta_ids: torch.Tensor,
        vol_ids: torch.Tensor,
        vb_ids: torch.Tensor,
        ind_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            delta_ids: (batch, seq_len) - delta token IDs
            vol_ids: (batch, seq_len) - volume token IDs
            vb_ids: (batch, seq_len) - volume bucket token IDs
            ind_dict: dict of 6 tensors, each (batch, seq_len)

        Returns:
            logits: (batch, num_classes)
            total_aux_loss: scalar
        """
        batch = delta_ids.shape[0]

        # Candle encoding
        delta_e = self.delta_emb(delta_ids)      # (batch, seq_len, 64)
        vol_e = self.vol_emb(vol_ids)             # (batch, seq_len, 16)
        vb_e = self.vb_emb(vb_ids)               # (batch, seq_len, 16)
        candle_cat = torch.cat([delta_e, vol_e, vb_e], dim=-1)  # (batch, seq_len, 96)
        candle_repr = self.candle_proj(candle_cat)  # (batch, seq_len, 128)

        # Indicator encoding
        ind_embs = [emb(ind_dict[key]) for emb, key in zip(
            self.indicator_embs,
            ["rsi", "macd_hist", "bollinger_pctb", "atr", "volume_ratio", "price_vs_sma"],
        )]
        ind_cat = torch.cat(ind_embs, dim=-1)    # (batch, seq_len, 96)
        ind_repr = self.indicator_proj(ind_cat)    # (batch, seq_len, 128)

        # Fusion
        x = torch.cat([candle_repr, ind_repr], dim=-1)  # (batch, seq_len, 256)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch, -1, -1)  # (batch, 1, 256)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, 256)

        # Add positional embeddings
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.pos_emb(positions).unsqueeze(0)

        # MoE Transformer layers
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss = total_aux_loss + aux_loss

        # CLS pooling: take position 0
        cls_out = x[:, 0, :]  # (batch, 256)

        # MLP head
        logits = self.head(cls_out)  # (batch, num_classes)

        return logits, total_aux_loss
