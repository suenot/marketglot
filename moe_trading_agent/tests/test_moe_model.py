"""Tests for MoETradingModel."""
import torch
from models.moe_model import MoETradingModel


def _make_batch(batch_size=4, seq_len=128):
    """Create a fake batch for testing."""
    delta_ids = torch.randint(0, 122, (batch_size, seq_len))
    vol_ids = torch.randint(0, 10, (batch_size, seq_len))
    vb_ids = torch.randint(0, 10, (batch_size, seq_len))
    ind_dict = {
        "rsi": torch.randint(0, 7, (batch_size, seq_len)),
        "macd_hist": torch.randint(0, 9, (batch_size, seq_len)),
        "bollinger_pctb": torch.randint(0, 7, (batch_size, seq_len)),
        "atr": torch.randint(0, 8, (batch_size, seq_len)),
        "volume_ratio": torch.randint(0, 7, (batch_size, seq_len)),
        "price_vs_sma": torch.randint(0, 7, (batch_size, seq_len)),
    }
    return delta_ids, vol_ids, vb_ids, ind_dict


def test_output_shape():
    """batch=4, seq_len=128 produces logits (4, 3), aux_loss scalar."""
    model = MoETradingModel(seq_len=128, num_experts=8, top_k=2, num_layers=4,
                            num_heads=8, dim=256, hidden_dim=1024, dropout=0.1, num_classes=3)
    delta_ids, vol_ids, vb_ids, ind_dict = _make_batch()
    logits, aux_loss = model(delta_ids, vol_ids, vb_ids, ind_dict)
    assert logits.shape == (4, 3), f"Expected (4, 3), got {logits.shape}"
    assert aux_loss.dim() == 0, f"Expected scalar aux_loss, got shape {aux_loss.shape}"


def test_logits_finite():
    """No NaN or Inf in logits."""
    model = MoETradingModel(seq_len=128, num_experts=8, top_k=2, num_layers=4,
                            num_heads=8, dim=256, hidden_dim=1024, dropout=0.1, num_classes=3)
    delta_ids, vol_ids, vb_ids, ind_dict = _make_batch()
    logits, _ = model(delta_ids, vol_ids, vb_ids, ind_dict)
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"


def test_gradients_flow():
    """Backward works through entire model."""
    model = MoETradingModel(seq_len=128, num_experts=8, top_k=2, num_layers=4,
                            num_heads=8, dim=256, hidden_dim=1024, dropout=0.1, num_classes=3)
    delta_ids, vol_ids, vb_ids, ind_dict = _make_batch()
    logits, aux_loss = model(delta_ids, vol_ids, vb_ids, ind_dict)
    loss = logits.sum() + aux_loss
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_cls_pooling():
    """Changing CLS token parameter changes the output."""
    model = MoETradingModel(seq_len=128, num_experts=8, top_k=2, num_layers=4,
                            num_heads=8, dim=256, hidden_dim=1024, dropout=0.1, num_classes=3)
    model.eval()

    delta_ids, vol_ids, vb_ids, ind_dict = _make_batch(batch_size=1)

    with torch.no_grad():
        logits1, _ = model(delta_ids, vol_ids, vb_ids, ind_dict)

    # Modify the CLS token parameter
    with torch.no_grad():
        original_cls = model.cls_token.clone()
        model.cls_token.data = torch.randn_like(model.cls_token)

    with torch.no_grad():
        logits2, _ = model(delta_ids, vol_ids, vb_ids, ind_dict)

    # Restore
    model.cls_token.data = original_cls

    assert not torch.allclose(logits1, logits2, atol=1e-4), \
        "Changing CLS token should change output"
