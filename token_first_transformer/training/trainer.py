from __future__ import annotations

import os
import math
import random
import tempfile
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
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


class ResumableRandomSampler(Sampler[int]):
    """Deterministic per-epoch shuffle that can start after a saved batch."""

    def __init__(self, data_source, seed: int = 42) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 1
        self.start_index = 0

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        if not 0 <= start_index <= len(self.data_source):
            raise ValueError("start_index exceeds dataset")
        self.epoch = epoch
        self.start_index = start_index

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(self.data_source), generator=generator)
        for index in permutation[self.start_index:]:
            yield int(index)

    def __len__(self) -> int:
        return len(self.data_source) - self.start_index


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
        seed: int = 42,
        checkpoint_every_steps: int = 100,
        resume_from: Path | None = None,
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
        self.seed = seed
        if checkpoint_every_steps < 1:
            raise ValueError("checkpoint_every_steps must be positive")
        self.checkpoint_every_steps = checkpoint_every_steps

        weights_tensor = None
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.next_epoch = 1
        self.next_batch = 0
        self.global_step = 0
        self.partial_loss = 0.0
        self.partial_batches = 0
        self.best_f1 = -1.0
        self.patience_counter = 0
        self.all_metrics: list[dict] = []
        self._pending_rng: dict | None = None
        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))

    @staticmethod
    def _auto_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def train(self) -> list[dict]:
        for epoch in range(self.next_epoch, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, val_f1 = self._val_epoch()

            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_f1": round(val_f1, 4),
            }
            self.all_metrics.append(metrics)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

            improved = val_f1 > self.best_f1
            if improved:
                self.best_f1 = val_f1
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.scheduler.step()
            self.next_epoch = epoch + 1
            self.next_batch = 0
            self.partial_loss = 0.0
            self.partial_batches = 0
            if improved:
                self._save_checkpoint("best.pt", epoch)
            self._save_checkpoint(f"epoch_{epoch}.pt", epoch)
            self._save_checkpoint("latest.pt", epoch)

            if self.patience_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        return self.all_metrics

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = self.partial_loss
        n_batches = self.partial_batches
        self.optimizer.zero_grad()
        sampler = self.train_loader.sampler
        resumable_sampler = isinstance(sampler, ResumableRandomSampler)
        if resumable_sampler:
            if self.train_loader.batch_size is None:
                raise ValueError("resumable sampler requires a fixed batch size")
            sampler.set_epoch(epoch, min(len(sampler.data_source), self.next_batch * self.train_loader.batch_size))
            total_batches = math.ceil(len(sampler.data_source) / self.train_loader.batch_size)
        else:
            generator = getattr(sampler, "generator", None)
            if generator is not None:
                generator.manual_seed(self.seed + epoch)
            total_batches = len(self.train_loader)
        if self.next_batch and not resumable_sampler and sampler.__class__.__name__ == "RandomSampler" and generator is None:
            raise ValueError("mid-epoch resume requires a seeded train DataLoader generator")

        for i, (delta, vol, vb, labels) in enumerate(self.train_loader, start=self.next_batch if resumable_sampler else 0):
            if not resumable_sampler and i < self.next_batch:
                continue
            if self._pending_rng is not None:
                self._restore_rng(self._pending_rng)
                self._pending_rng = None
            delta = delta.to(self.device)
            vol = vol.to(self.device)
            vb = vb.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(delta, vol, vb)
            remaining = total_batches - (i // self.grad_accum_steps) * self.grad_accum_steps
            accumulation_size = min(self.grad_accum_steps, remaining)
            loss = self.criterion(logits, labels) / accumulation_size
            loss.backward()

            optimizer_step = (i + 1) % self.grad_accum_steps == 0 or i + 1 == total_batches
            if optimizer_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * accumulation_size
            n_batches += 1
            self.next_batch = i + 1
            self.partial_loss = total_loss
            self.partial_batches = n_batches
            if optimizer_step and self.global_step % self.checkpoint_every_steps == 0:
                self._save_checkpoint("latest.pt", epoch)

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
        state = {
            "epoch": epoch,
            "next_epoch": self.next_epoch,
            "next_batch": self.next_batch,
            "global_step": self.global_step,
            "partial_loss": self.partial_loss,
            "partial_batches": self.partial_batches,
            "best_f1": self.best_f1,
            "patience_counter": self.patience_counter,
            "all_metrics": self.all_metrics,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "rng": self._capture_rng(),
        }
        # A power loss must leave either the previous complete checkpoint or this one.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{name}.", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                torch.save(state, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
                os.replace(tmp_path, path)
            finally:
                tmp_path.unlink(missing_ok=True)

    def _load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.next_epoch = int(state.get("next_epoch", int(state["epoch"]) + 1))
        self.next_batch = int(state.get("next_batch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.partial_loss = float(state.get("partial_loss", 0.0))
        self.partial_batches = int(state.get("partial_batches", 0))
        self.best_f1 = float(state.get("best_f1", -1.0))
        self.patience_counter = int(state.get("patience_counter", 0))
        self.all_metrics = list(state.get("all_metrics", []))
        self._pending_rng = state.get("rng")

    @staticmethod
    def _capture_rng() -> dict:
        state = {"python": random.getstate(), "torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state") and torch.backends.mps.is_available():
            state["mps"] = torch.mps.get_rng_state()
        return state

    @staticmethod
    def _restore_rng(state: dict) -> None:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        if "mps" in state and torch.backends.mps.is_available():
            torch.mps.set_rng_state(state["mps"])
