from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from models.multimodal_model import MultimodalEncoder


class Trainer:
    def __init__(self, model: MultimodalEncoder, train_loader: DataLoader, val_loader: DataLoader,
                 epochs=10, lr=3e-4, weight_decay=0.01, early_stop_patience=3,
                 device="auto", checkpoint_dir=Path("checkpoints")):
        self.device = ("mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.ckpt_dir = Path(checkpoint_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

    def train(self):
        best_f1, pat = -1.0, 0
        for ep in range(1, self.epochs + 1):
            tl = self._train_ep()
            vl, vf = self._val_ep()
            print(f"Ep {ep}: train={tl:.4f} val={vl:.4f} f1={vf:.4f}")
            if vf > best_f1:
                best_f1, pat = vf, 0
                torch.save({"model": self.model.state_dict()}, self.ckpt_dir / "best.pt")
            else:
                pat += 1
            torch.save({"model": self.model.state_dict()}, self.ckpt_dir / f"ep{ep}.pt")
            if pat >= self.early_stop_patience:
                print(f"Early stop at ep {ep}"); break
            self.scheduler.step()

    def _train_ep(self):
        self.model.train(); tl, n = 0.0, 0
        for b in self.train_loader:
            delta, vol, vb, ind, y = b
            dev = self.device
            logits = self.model(delta.to(dev), vol.to(dev), vb.to(dev),
                                [t.to(dev) for t in ind])
            loss = self.criterion(logits, y.to(dev))
            loss.backward(); self.optimizer.step(); self.optimizer.zero_grad()
            tl += loss.item(); n += 1
        return tl / max(n, 1)

    def _val_ep(self):
        self.model.eval(); tl, n = 0.0, 0; preds, labels = [], []
        with torch.no_grad():
            for b in self.val_loader:
                delta, vol, vb, ind, y = b
                logits = self.model(delta.to(self.device), vol.to(self.device), vb.to(self.device),
                                    [t.to(self.device) for t in ind])
                tl += self.criterion(logits, y.to(self.device)).item(); n += 1
                preds.extend(logits.argmax(-1).cpu().numpy())
                labels.extend(y.numpy())
        return tl/max(n,1), float(f1_score(labels, preds, average="weighted", zero_division=0))
