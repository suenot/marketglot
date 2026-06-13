import torch

from models.orderbook_mlp import OrderbookEncoder, OrderbookClassifier


def _make_classifier(input_dim=80, embedding_dim=64, num_classes=3):
    encoder = OrderbookEncoder(
        input_dim=input_dim, hidden_dims=[256, 128],
        embedding_dim=embedding_dim, dropout=0.1,
    )
    return OrderbookClassifier(encoder, num_classes=num_classes)


def test_encoder_output_shape():
    encoder = OrderbookEncoder(input_dim=80, hidden_dims=[128, 64], embedding_dim=32)
    x = torch.randn(8, 80)
    emb = encoder(x)
    assert emb.shape == (8, 32)


def test_classifier_forward_shape():
    model = _make_classifier()
    x = torch.randn(4, 80)
    logits = model(x)
    assert logits.shape == (4, 3)


def test_classifier_encode_shape():
    model = _make_classifier(embedding_dim=64)
    x = torch.randn(5, 80)
    emb = model.encode(x)
    assert emb.shape == (5, 64)


def test_single_input():
    model = _make_classifier()
    x = torch.randn(1, 80)
    assert model(x).shape == (1, 3)


def test_gradients_flow():
    model = _make_classifier()
    x = torch.randn(3, 80)
    model(x).sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_output_finite():
    model = _make_classifier()
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 80))
    assert torch.isfinite(logits).all()
