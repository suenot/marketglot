"""Tests for Expert module."""
import torch
from models.expert import Expert


def test_output_shape():
    """Input (32, 256) produces output (32, 256)."""
    expert = Expert(dim=256, hidden_dim=1024)
    x = torch.randn(32, 256)
    out = expert(x)
    assert out.shape == (32, 256), f"Expected (32, 256), got {out.shape}"


def test_gradients_flow():
    """Gradients exist on all parameters after backward."""
    expert = Expert(dim=256, hidden_dim=1024)
    x = torch.randn(32, 256, requires_grad=True)
    out = expert(x)
    loss = out.sum()
    loss.backward()
    for name, param in expert.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_finite_output():
    """Output contains no NaN or Inf values."""
    expert = Expert(dim=256, hidden_dim=1024)
    x = torch.randn(32, 256)
    out = expert(x)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"
