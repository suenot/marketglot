import torch
import pytest
from models.price_transformer import PriceTransformer


def test_output_shape():
    model = PriceTransformer(
        delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=64, bucket_emb_dim=16,
        hidden_dim=256, num_layers=4, num_heads=8,
        ffn_dim=1024, dropout=0.1, num_classes=3, seq_len=128,
    )
    batch = 4
    delta = torch.randint(2, 121, (batch, 128))
    vol = torch.randint(2, 9, (batch, 128))
    vb = torch.randint(2, 9, (batch, 128))
    logits = model(delta, vol, vb)
    assert logits.shape == (batch, 3)


def test_single_input():
    model = PriceTransformer(
        delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=64, bucket_emb_dim=16,
        hidden_dim=256, num_layers=2, num_heads=4,
        ffn_dim=512, dropout=0.0, num_classes=3, seq_len=64,
    )
    delta = torch.randint(2, 121, (1, 64))
    vol = torch.randint(2, 9, (1, 64))
    vb = torch.randint(2, 9, (1, 64))
    logits = model(delta, vol, vb)
    assert logits.shape == (1, 3)


def test_gradients_flow():
    model = PriceTransformer(
        delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=32, bucket_emb_dim=8,
        hidden_dim=64, num_layers=1, num_heads=2,
        ffn_dim=128, dropout=0.0, num_classes=3, seq_len=16,
    )
    delta = torch.randint(2, 121, (2, 16))
    vol = torch.randint(2, 9, (2, 16))
    vb = torch.randint(2, 9, (2, 16))
    logits = model(delta, vol, vb)
    logits.sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_cls_position_output_finite():
    model = PriceTransformer(
        delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=32, bucket_emb_dim=8,
        hidden_dim=64, num_layers=1, num_heads=2,
        ffn_dim=128, dropout=0.0, num_classes=3, seq_len=16,
    )
    model.eval()
    with torch.no_grad():
        delta = torch.randint(2, 121, (1, 16))
        vol = torch.randint(2, 9, (1, 16))
        vb = torch.randint(2, 9, (1, 16))
        logits = model(delta, vol, vb)
    assert torch.isfinite(logits).all()
