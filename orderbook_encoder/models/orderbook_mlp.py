"""Deep MLP encoder for L2 order book feature vectors.

OrderbookEncoder maps a flat feature vector to a compact embedding through a
stack of Linear -> LayerNorm -> GELU -> Dropout blocks. OrderbookClassifier
wraps the encoder with a linear head for the 3-class movement target.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OrderbookEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        embedding_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.net = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OrderbookClassifier(nn.Module):
    def __init__(self, encoder: OrderbookEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
