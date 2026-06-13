from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from models.moe_model import MoETradingModel


def get_device() -> torch.device:
    """Auto-detect best available device: MPS > CUDA > CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class Trainer:
    """Trainer for MoETradingModel."""

    def __init__(
        self,
        model: MoETradingModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        aux_loss_lambda: float = 0.01,
        patience: int = 3,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.aux_loss_lambda = aux_loss_lambda
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Compute class weights from training data
        self.class_weights = self._compute_class_weights().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.best_val_f1 = 0.0
        self.patience_counter = 0

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights from training data."""
        counts = torch.zeros(3)
        for batch in self.train_loader:
            labels = batch[-1]  # Last element is label
            for c in range(3):
                counts[c] += (labels == c).sum().item()
        # Inverse frequency weighting
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * 3  # Normalize
        return weights.float()

    def _train_epoch(self) -> float:
        """Train one epoch, return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            delta_ids, vol_ids, vb_ids, ind_dict, labels = batch
            delta_ids = delta_ids.to(self.device)
            vol_ids = vol_ids.to(self.device)
            vb_ids = vb_ids.to(self.device)
            ind_dict = {k: v.to(self.device) for k, v in ind_dict.items()}
            labels = labels.to(self.device)

            logits, aux_loss = self.model(delta_ids, vol_ids, vb_ids, ind_dict)
            ce_loss = self.criterion(logits, labels)
            loss = ce_loss + self.aux_loss_lambda * aux_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(self) -> tuple[float, float]:
        """Validate one epoch, return (loss, weighted F1)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                delta_ids, vol_ids, vb_ids, ind_dict, labels = batch
                delta_ids = delta_ids.to(self.device)
                vol_ids = vol_ids.to(self.device)
                vb_ids = vb_ids.to(self.device)
                ind_dict = {k: v.to(self.device) for k, v in ind_dict.items()}
                labels = labels.to(self.device)

                logits, aux_loss = self.model(delta_ids, vol_ids, vb_ids, ind_dict)
                ce_loss = self.criterion(logits, labels)
                loss = ce_loss + self.aux_loss_lambda * aux_loss

                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return avg_loss, weighted_f1

    def train(self, epochs: int, scheduler: bool = True) -> None:
        """Full training loop with early stopping."""
        if scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        else:
            lr_scheduler = None

        print(f"Training on device: {self.device}")
        print(f"Class weights: {self.class_weights}")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_f1 = self._val_epoch()

            if lr_scheduler is not None:
                lr_scheduler.step()

            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val F1: {val_f1:.4f}"
            )

            # Early stopping and checkpoint
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_f1)
                print(f"  -> Saved best checkpoint (F1: {val_f1:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"  -> Early stopping at epoch {epoch}")
                    break

        print(f"Training complete. Best Val F1: {self.best_val_f1:.4f}")

    def _save_checkpoint(self, epoch: int, val_f1: float) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / "best_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_f1": val_f1,
        }, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with F1 {checkpoint['val_f1']:.4f}")
