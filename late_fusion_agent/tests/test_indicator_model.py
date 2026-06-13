import torch
import pytest
from models.indicator_model import IndicatorModel

VS = [7, 9, 7, 8, 7, 7]

def test_output_shape():
    model = IndicatorModel(vocab_sizes=VS, emb_dim=16, hidden_dim=128, num_layers=2, num_heads=4, ffn_dim=256, dropout=0.1, num_classes=3, seq_len=128)
    inputs = [torch.randint(2, vs - 1, (4, 128)) for vs in VS]
    assert model(inputs).shape == (4, 3)

def test_single_input():
    model = IndicatorModel(vocab_sizes=VS, emb_dim=8, hidden_dim=32, num_layers=1, num_heads=2, ffn_dim=64, dropout=0.0, num_classes=3, seq_len=32)
    inputs = [torch.randint(2, vs - 1, (1, 32)) for vs in VS]
    assert model(inputs).shape == (1, 3)

def test_gradients_flow():
    model = IndicatorModel(vocab_sizes=VS, emb_dim=8, hidden_dim=32, num_layers=1, num_heads=2, ffn_dim=64, dropout=0.0, num_classes=3, seq_len=16)
    inputs = [torch.randint(2, vs - 1, (2, 16)) for vs in VS]
    model(inputs).sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"

def test_output_finite():
    model = IndicatorModel(vocab_sizes=VS, emb_dim=8, hidden_dim=32, num_layers=1, num_heads=2, ffn_dim=64, dropout=0.0, num_classes=3, seq_len=16)
    model.eval()
    with torch.no_grad():
        inputs = [torch.randint(2, vs - 1, (1, 16)) for vs in VS]
        assert torch.isfinite(model(inputs)).all()
