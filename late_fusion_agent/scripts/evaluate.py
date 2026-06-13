"""Evaluate late-fusion pipeline on test set."""
from __future__ import annotations
import importlib
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Import PriceTransformer from sibling project via importlib to avoid
# namespace collision with our own models/ package.
_TFT_PATH = _PROJECT_ROOT.parent / "token_first_transformer" / "models" / "price_transformer.py"
_spec = importlib.util.spec_from_file_location("price_transformer", str(_TFT_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PriceTransformer = _mod.PriceTransformer

import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from dataset.fusion_dataset import FusionDataset
from models.indicator_model import IndicatorModel
from models.meta_model import MetaModel


def collate_fn(batch):
    delta = torch.stack([torch.tensor(b[0]) for b in batch])
    vol = torch.stack([torch.tensor(b[1]) for b in batch])
    vb = torch.stack([torch.tensor(b[2]) for b in batch])
    keys = list(batch[0][3].keys())
    ind = [torch.stack([torch.tensor(b[3][k]) for b in batch]) for k in keys]
    labels = torch.tensor([b[4] for b in batch])
    return delta, vol, vb, ind, labels


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    klines = data_dir / cfg["data"]["symbol"] / "klines_1m"
    s, e = cfg["data"]["test_months"]
    test_files = sorted([f for f in klines.glob("*.parquet") if s <= f.stem <= e])
    print(f"Test: {len(test_files)} files")

    sc, tc = cfg["sequence"], cfg["tokenizer"]
    test_ds = FusionDataset(test_files, seq_len=sc["length"], target_horizon=sc["target_horizon"], target_threshold=sc["target_threshold"], range_pct=tc["delta"]["range_pct"], step_pct=tc["delta"]["step_pct"], n_bins=tc["bucket"]["n_bins"])
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)

    ckpt = Path(args.checkpoint_dir)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    ma = cfg["model_a"]
    model_a = PriceTransformer(delta_vocab_size=ma["delta_vocab_size"], bucket_vocab_size=ma["bucket_vocab_size"], delta_emb_dim=ma["delta_emb_dim"], bucket_emb_dim=ma["bucket_emb_dim"], hidden_dim=ma["hidden_dim"], num_layers=ma["num_layers"], num_heads=ma["num_heads"], ffn_dim=ma["ffn_dim"], dropout=0.0, num_classes=ma["num_classes"], seq_len=sc["length"])
    model_a.load_state_dict(torch.load(ckpt / "model_a_best.pt", map_location="cpu", weights_only=True))
    model_a.to(dev).eval()

    mb = cfg["model_b"]
    model_b = IndicatorModel(vocab_sizes=mb["vocab_sizes"], emb_dim=mb["emb_dim"], hidden_dim=mb["hidden_dim"], num_layers=mb["num_layers"], num_heads=mb["num_heads"], ffn_dim=mb["ffn_dim"], dropout=0.0, num_classes=mb["num_classes"], seq_len=sc["length"])
    model_b.load_state_dict(torch.load(ckpt / "model_b_best.pt", map_location="cpu", weights_only=True))
    model_b.to(dev).eval()

    meta = MetaModel()
    meta.load_state_dict(torch.load(ckpt / "meta_model.pt", map_location="cpu", weights_only=True))
    meta.to(dev).eval()

    preds, labels = [], []
    with torch.no_grad():
        for delta, vol, vb, ind, y in test_dl:
            la = model_a(delta.to(dev), vol.to(dev), vb.to(dev))
            lb = model_b([t.to(dev) for t in ind])
            fused = meta(la, lb)
            preds.extend(fused.argmax(-1).cpu().numpy())
            labels.extend(y.numpy())

    print("\n=== Late Fusion Results ===")
    print(classification_report(labels, preds, target_names=["DOWN", "FLAT", "UP"]))
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
