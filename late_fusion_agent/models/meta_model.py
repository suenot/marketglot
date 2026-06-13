from __future__ import annotations
import torch
import torch.nn as nn


class MetaModel(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 16, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([logits_a, logits_b], dim=-1))
