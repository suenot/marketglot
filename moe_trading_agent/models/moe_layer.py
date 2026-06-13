from __future__ import annotations

import torch
import torch.nn as nn

from models.expert import Expert
from models.router import Router


class MoELayer(nn.Module):
    """Mixture of Experts layer: router + sparse expert dispatch."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [Expert(dim, hidden_dim) for _ in range(num_experts)]
        )
        self.router = Router(dim, num_experts, top_k)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with sparse expert routing.

        Args:
            x: (batch, seq_len, dim)

        Returns:
            output: (batch, seq_len, dim)
            aux_loss: scalar
        """
        batch, seq_len, dim = x.shape
        # Reshape to (batch*seq_len, dim)
        flat_x = x.reshape(-1, dim)
        num_tokens = flat_x.shape[0]

        # Get routing decisions
        top_k_indices, top_k_weights, aux_loss = self.router(flat_x)
        # top_k_indices: (num_tokens, top_k), top_k_weights: (num_tokens, top_k)

        # Initialize output
        output = torch.zeros_like(flat_x)

        # Sparse dispatch: for each expert, gather tokens assigned to it
        for k in range(self.top_k):
            # expert_indices: (num_tokens,) - which expert for each token at this k
            expert_indices = top_k_indices[:, k]
            # weights: (num_tokens,) - weight for each token at this k
            weights = top_k_weights[:, k]

            for expert_idx in range(self.num_experts):
                # Find tokens routed to this expert at position k
                mask = (expert_indices == expert_idx)
                if not mask.any():
                    continue

                # Gather tokens for this expert
                expert_input = flat_x[mask]  # (n_tokens_for_expert, dim)
                expert_output = self.experts[expert_idx](expert_input)  # (n_tokens_for_expert, dim)

                # Scatter back with weights
                expert_weights = weights[mask].unsqueeze(-1)  # (n_tokens_for_expert, 1)
                output[mask] += expert_weights * expert_output

        # Reshape back to (batch, seq_len, dim)
        output = output.reshape(batch, seq_len, dim)

        return output, aux_loss
