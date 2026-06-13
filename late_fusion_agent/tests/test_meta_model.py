import torch
import pytest
from models.meta_model import MetaModel

def test_output_shape():
    m = MetaModel(input_dim=6, hidden_dim=16, num_classes=3)
    assert m(torch.randn(8, 3), torch.randn(8, 3)).shape == (8, 3)

def test_single_input():
    m = MetaModel(input_dim=6, hidden_dim=16, num_classes=3)
    assert m(torch.randn(1, 3), torch.randn(1, 3)).shape == (1, 3)

def test_gradients():
    m = MetaModel(input_dim=6, hidden_dim=16, num_classes=3)
    m(torch.randn(2, 3), torch.randn(2, 3)).sum().backward()
    for p in m.parameters():
        assert p.grad is not None

def test_output_finite():
    m = MetaModel(input_dim=6, hidden_dim=16, num_classes=3)
    m.eval()
    with torch.no_grad():
        assert torch.isfinite(m(torch.randn(1, 3), torch.randn(1, 3))).all()
