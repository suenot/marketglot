from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Top-K gating router with auxiliary load-balancing loss."""

    def __init__(self, dim: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            x: (num_tokens, dim)

        Returns:
            top_k_indices: (num_tokens, top_k) - expert indices
            top_k_weights: (num_tokens, top_k) - gate weights (sum to 1 over top_k)
            aux_loss: scalar - load balancing auxiliary loss
        """
        logits = self.gate(x)  # (num_tokens, num_experts)
        probs = F.softmax(logits, dim=-1)  # (num_tokens, num_experts)

        # Top-K selection
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        # Normalize weights so they sum to 1 over top_k
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-10)

        # Auxiliary load-balancing loss
        # f_i = fraction of tokens routed to expert i (count-based)
        # p_i = mean gate probability for expert i
        # aux_loss = num_experts * sum(f_i * p_i)
        num_tokens = x.shape[0]
        # Count how many tokens are routed to each expert
        one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts).float()  # (num_tokens, top_k, num_experts)
        expert_mask = one_hot.sum(dim=1)  # (num_tokens, num_experts)
        f = expert_mask.mean(dim=0)  # (num_experts,)
        p = probs.mean(dim=0)  # (num_experts,)
        aux_loss = self.num_experts * torch.sum(f * p)

        return top_k_indices, top_k_weights, aux_loss
