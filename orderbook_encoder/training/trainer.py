"""Training loop for the order book MLP classifier.

``train(cfg)`` builds the train/val/test splits, trains the classifier with
AdamW and class-weighted cross-entropy, early-stops on validation loss, and
writes artifacts to ``{artifacts_dir}/run_YYYYMMDD_HHMMSS/``:
    best.pt            best model state dict (by val loss)
    config.json        the config used for the run
    test_metrics.json  classification report + confusion matrix on the test set
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset.orderbook_dataset import OrderbookDataset, build_splits
from models.orderbook_mlp import OrderbookEncoder, OrderbookClassifier

TARGET_NAMES = ["DOWN", "FLAT", "UP"]


def compute_class_weights(labels: list[int], num_classes: int = 3) -> list[float]:
    """Weights inversely proportional to class frequency in ``labels``."""
    counts = Counter(labels)
    total = max(len(labels), 1)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        weights.append(total / (num_classes * cnt))
    return weights


def _auto_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _train_labels(ds: OrderbookDataset) -> list[int]:
    return [int(ds[i][1]) for i in range(len(ds))]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: AdamW | None = None,
) -> tuple[float, list[int], list[int]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    n_batches = 0
    preds: list[int] = []
    labels: list[int] = []

    with torch.set_grad_enabled(train_mode):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

    return total_loss / max(n_batches, 1), preds, labels


def train(cfg: dict) -> dict:
    train_ds, val_ds, test_ds = build_splits(cfg)
    print(f"Datasets: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    tcfg = cfg["training"]
    mcfg = cfg["model"]
    device = _auto_device(tcfg["device"])
    batch_size = tcfg["batch_size"]

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    encoder = OrderbookEncoder(
        input_dim=mcfg["input_dim"],
        hidden_dims=mcfg["hidden_dims"],
        embedding_dim=mcfg["embedding_dim"],
        dropout=mcfg["dropout"],
    )
    model = OrderbookClassifier(encoder, num_classes=mcfg["num_classes"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    class_weights = compute_class_weights(_train_labels(train_ds), mcfg["num_classes"])
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(
        model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"]
    )

    run_dir = Path(tcfg["artifacts_dir"]) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    patience = 0
    for epoch in range(1, tcfg["epochs"] + 1):
        train_loss, _, _ = _run_epoch(model, train_dl, criterion, device, optimizer)
        val_loss, _, _ = _run_epoch(model, val_dl, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       run_dir / "best.pt")
        else:
            patience += 1
            if patience >= tcfg["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Evaluate the best checkpoint on the test set.
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])

    if len(test_ds) == 0:
        print("Warning: empty test split, skipping test metrics")
        report: dict = {}
        cm = np.zeros((3, 3), dtype=int)
    else:
        _, test_preds, test_labels = _run_epoch(model, test_dl, criterion, device)
        report = classification_report(
            test_labels, test_preds, target_names=TARGET_NAMES,
            labels=[0, 1, 2], output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(test_labels, test_preds, labels=[0, 1, 2])

    metrics = {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "best_val_loss": best_val,
    }
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    metrics["run_dir"] = str(run_dir)
    return metrics
