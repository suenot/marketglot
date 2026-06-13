from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Import PriceTransformer from sibling project via importlib to avoid
# namespace collision with our own models/ package.
_TFT_MODELS_PATH = _PROJECT_ROOT / "token_first_transformer" / "models" / "price_transformer.py"
_spec = importlib.util.spec_from_file_location("price_transformer", str(_TFT_MODELS_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PriceTransformer = _mod.PriceTransformer

from models.indicator_model import IndicatorModel
from models.meta_model import MetaModel


class FusionTrainer:
    def __init__(self, model_a: PriceTransformer, model_b: IndicatorModel,
                 train_loader: DataLoader, val_loader: DataLoader,
                 epochs_a=5, epochs_b=5, epochs_meta=10,
                 lr=3e-4, weight_decay=0.01, early_stop_patience=3,
                 device="auto", checkpoint_dir=Path("checkpoints")):
        self.device = self._auto_device() if device == "auto" else device
        self.model_a = model_a.to(self.device)
        self.model_b = model_b.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs_a = epochs_a
        self.epochs_b = epochs_b
        self.epochs_meta = epochs_meta
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stop_patience = early_stop_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _auto_device():
        if torch.backends.mps.is_available(): return "mps"
        if torch.cuda.is_available(): return "cuda"
        return "cpu"

    def train_all(self):
        print("=== Training Model A ===")
        self._train_base(self.model_a, self.epochs_a, "model_a",
            lambda b: self.model_a(b[0].to(self.device), b[1].to(self.device), b[2].to(self.device)))

        print("\n=== Training Model B ===")
        self._train_base(self.model_b, self.epochs_b, "model_b",
            lambda b: self.model_b([t.to(self.device) for t in b[3]]))

        print("\n=== Collecting val logits ===")
        la, lb, y = self._collect_logits()

        print("\n=== Training Meta-Model ===")
        meta = MetaModel().to(self.device)
        self._train_meta(meta, la, lb, y)
        torch.save(meta.state_dict(), self.checkpoint_dir / "meta_model.pt")
        return meta

    def _train_base(self, model, epochs, name, forward_fn):
        opt = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = nn.CrossEntropyLoss()
        best, pat = float("inf"), 0
        for ep in range(1, epochs + 1):
            model.train()
            tl, n = 0.0, 0
            for b in self.train_loader:
                loss = crit(forward_fn(b), b[4].to(self.device))
                loss.backward(); opt.step(); opt.zero_grad()
                tl += loss.item(); n += 1
            vl, vn = 0.0, 0
            model.eval()
            with torch.no_grad():
                for b in self.val_loader:
                    vl += crit(forward_fn(b), b[4].to(self.device)).item(); vn += 1
            vl /= max(vn, 1)
            print(f"  {name} ep {ep}: train={tl/max(n,1):.4f} val={vl:.4f}")
            if vl < best:
                best, pat = vl, 0
                torch.save(model.state_dict(), self.checkpoint_dir / f"{name}_best.pt")
            else:
                pat += 1
            if pat >= self.early_stop_patience:
                print(f"  Early stop {name} at ep {ep}"); break

    def _collect_logits(self):
        self.model_a.eval(); self.model_b.eval()
        la, lb, y = [], [], []
        with torch.no_grad():
            for b in self.val_loader:
                la.append(self.model_a(b[0].to(self.device), b[1].to(self.device), b[2].to(self.device)).cpu())
                lb.append(self.model_b([t.to(self.device) for t in b[3]]).cpu())
                y.append(b[4])
        return torch.cat(la), torch.cat(lb), torch.cat(y)

    def _train_meta(self, meta, la, lb, y):
        opt = AdamW(meta.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = nn.CrossEntropyLoss()
        ds = torch.utils.data.TensorDataset(la, lb, y)
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        for ep in range(1, self.epochs_meta + 1):
            meta.train(); tl = 0.0
            for a, b, lbl in dl:
                loss = crit(meta(a.to(self.device), b.to(self.device)), lbl.to(self.device))
                loss.backward(); opt.step(); opt.zero_grad()
                tl += loss.item()
            print(f"  meta ep {ep}: loss={tl/len(dl):.4f}")
