from __future__ import annotations

import torch
import torch.nn as nn


class Expert(nn.Module):
    """Single expert FFN: Linear -> ReLU -> Linear."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (num_tokens, dim) -> (num_tokens, dim)"""
        return self.net(x)
