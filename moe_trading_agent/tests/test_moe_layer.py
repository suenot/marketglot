"""Tests for MoELayer module."""
import torch
from models.moe_layer import MoELayer


def test_output_shape():
    """(batch, seq, dim) -> (batch, seq, dim)."""
    layer = MoELayer(dim=256, hidden_dim=1024, num_experts=8, top_k=2)
    x = torch.randn(4, 16, 256)
    out, aux_loss = layer(x)
    assert out.shape == (4, 16, 256), f"Expected (4, 16, 256), got {out.shape}"


def test_aux_loss_returned():
    """aux_loss is a scalar > 0."""
    layer = MoELayer(dim=256, hidden_dim=1024, num_experts=8, top_k=2)
    x = torch.randn(4, 16, 256)
    _, aux_loss = layer(x)
    assert aux_loss.dim() == 0, f"Expected scalar, got shape {aux_loss.shape}"
    assert aux_loss.item() > 0, f"Expected positive aux_loss, got {aux_loss.item()}"


def test_residual_compatible():
    """Output can be added to input (same shape)."""
    layer = MoELayer(dim=256, hidden_dim=1024, num_experts=8, top_k=2)
    x = torch.randn(4, 16, 256)
    out, _ = layer(x)
    residual = x + out
    assert residual.shape == (4, 16, 256)


def test_different_top_k():
    """Works with top_k=1 and top_k=3."""
    for top_k in [1, 3]:
        layer = MoELayer(dim=256, hidden_dim=1024, num_experts=8, top_k=top_k)
        x = torch.randn(4, 16, 256)
        out, aux_loss = layer(x)
        assert out.shape == (4, 16, 256), f"Failed with top_k={top_k}"
        assert aux_loss.item() > 0, f"aux_loss not positive with top_k={top_k}"
