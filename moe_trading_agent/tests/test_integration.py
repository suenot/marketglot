"""Integration test: mock parquet -> dataset -> model forward -> loss backward."""
import os
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def _create_mock_parquet(tmpdir, n_rows=500, base_price=30000.0):
    """Create mock parquet with realistic-looking BTC price data."""
    np.random.seed(42)
    returns = np.random.randn(n_rows) * 0.001
    close = base_price * np.exp(np.cumsum(returns)).astype(np.float32)
    high = (close * (1 + np.abs(np.random.randn(n_rows)) * 0.001)).astype(np.float32)
    low = (close * (1 - np.abs(np.random.randn(n_rows)) * 0.001)).astype(np.float32)
    open_prices = (close + np.random.randn(n_rows) * 5).astype(np.float32)
    volume = (np.abs(np.random.randn(n_rows)) * 100 + 50).astype(np.float32)
    timestamp = np.arange(n_rows, dtype=np.int32)

    df = pd.DataFrame({
        "timestamp": timestamp,
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    parquet_path = os.path.join(tmpdir, "mock_data.parquet")
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def test_full_pipeline():
    """Mock parquet -> dataset -> model forward -> loss backward -> verify shapes."""
    from dataset.moe_dataset import MoEDataset
    from models.moe_model import MoETradingModel

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = _create_mock_parquet(tmpdir)

        # Create dataset
        dataset = MoEDataset(
            file_paths=[parquet_path],
            seq_len=128,
            horizon=60,
            threshold=0.0015,
        )

        # Create dataloader
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        delta_ids, vol_ids, vb_ids, ind_dict, labels = batch

        # Verify batch shapes
        assert delta_ids.shape == (4, 128)
        assert labels.shape == (4,)

        # Create model (smaller for test speed)
        model = MoETradingModel(
            seq_len=128, num_experts=4, top_k=2, num_layers=2,
            num_heads=4, dim=256, hidden_dim=512, dropout=0.1, num_classes=3,
        )

        # Forward pass
        logits, aux_loss = model(delta_ids, vol_ids, vb_ids, ind_dict)
        assert logits.shape == (4, 3), f"Expected logits (4, 3), got {logits.shape}"
        assert aux_loss.dim() == 0, f"aux_loss should be scalar"
        assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"

        # Compute loss
        ce_loss = torch.nn.functional.cross_entropy(logits, labels)
        loss = ce_loss + 0.01 * aux_loss
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

        # Backward pass
        loss.backward()

        # Verify gradients exist
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        assert grad_count > 0, "No gradients computed"
