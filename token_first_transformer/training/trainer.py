from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import f1_score

from models.price_transformer import PriceTransformer


def compute_class_weights(labels: list[int], num_classes: int = 3) -> list[float]:
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        weights.append(total / (num_classes * cnt))
    return weights


class Trainer:
    def __init__(
        self,
        model: PriceTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: list[float] | None = None,
        epochs: int = 10,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        grad_accum_steps: int = 1,
        early_stop_patience: int = 3,
        device: str = "auto",
        checkpoint_dir: Path = Path("checkpoints"),
        max_threads: int = 4,
    ) -> None:
        torch.set_num_threads(max_threads)
        if torch.backends.mps.is_available():
            torch.set_num_threads(max_threads)
        if device == "auto":
            self.device = self._auto_device()
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps
        self.early_stop_patience = early_stop_patience
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        weights_tensor = None
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

    @staticmethod
    def _auto_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def train(self) -> list[dict]:
        best_f1 = -1.0
        patience_counter = 0
        all_metrics = []

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_f1 = self._val_epoch()

            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_f1": round(val_f1, 4),
            }
            all_metrics.append(metrics)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                self._save_checkpoint("best.pt", epoch)
            else:
                patience_counter += 1

            self._save_checkpoint(f"epoch_{epoch}.pt", epoch)

            if patience_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

            self.scheduler.step()

        return all_metrics

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for i, (delta, vol, vb, labels) in enumerate(self.train_loader):
            delta = delta.to(self.device)
            vol = vol.to(self.device)
            vb = vb.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(delta, vol, vb)
            loss = self.criterion(logits, labels) / self.grad_accum_steps
            loss.backward()

            if (i + 1) % self.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for delta, vol, vb, labels in self.val_loader:
                delta = delta.to(self.device)
                vol = vol.to(self.device)
                vb = vb.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(delta, vol, vb)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return avg_loss, float(f1)

    def _save_checkpoint(self, name: str, epoch: int) -> None:
        path = self.checkpoint_dir / name
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
