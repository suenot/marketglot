"""Evaluate multimodal encoder on test set."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "token_first_transformer"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "indicator_tokenizer"))

import yaml, torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from dataset.multimodal_dataset import MultimodalDataset
from models.multimodal_model import MultimodalEncoder


def collate(batch):
    delta = torch.stack([torch.tensor(b[0]) for b in batch])
    vol = torch.stack([torch.tensor(b[1]) for b in batch])
    vb = torch.stack([torch.tensor(b[2]) for b in batch])
    keys = list(batch[0][3].keys())
    ind = [torch.stack([torch.tensor(b[3][k]) for b in batch]) for k in keys]
    y = torch.tensor([b[4] for b in batch])
    return delta, vol, vb, ind, y


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = p.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)

    kd = Path(cfg["data"]["data_dir"]) / cfg["data"]["symbol"] / "klines_1m"
    s,e = cfg["data"]["test_months"]
    te_f = sorted([f for f in kd.glob("*.parquet") if s <= f.stem <= e])
    print(f"Test: {len(te_f)}")

    sc, tc = cfg["sequence"], cfg["tokenizer"]
    mk = dict(seq_len=sc["length"], target_horizon=sc["target_horizon"], target_threshold=sc["target_threshold"],
              range_pct=tc["delta"]["range_pct"], step_pct=tc["delta"]["step_pct"], n_bins=tc["bucket"]["n_bins"])
    te_dl = DataLoader(MultimodalDataset(te_f, **mk), batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)

    mc, mi, mf = cfg["model"]["candle"], cfg["model"]["indicator"], cfg["model"]["fusion"]
    model = MultimodalEncoder(
        delta_vocab_size=mc["delta_vocab_size"], bucket_vocab_size=mc["bucket_vocab_size"],
        delta_emb_dim=mc["delta_emb_dim"], bucket_emb_dim=mc["bucket_emb_dim"], candle_proj_dim=mc["proj_dim"],
        ind_vocab_sizes=mi["vocab_sizes"], ind_emb_dim=mi["emb_dim"], ind_proj_dim=mi["proj_dim"],
        hidden_dim=mf["hidden_dim"], num_layers=mf["num_layers"], num_heads=mf["num_heads"],
        ffn_dim=mf["ffn_dim"], dropout=0.0, num_classes=mf["num_classes"], seq_len=sc["length"])

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev).eval()

    preds, labels = [], []
    with torch.no_grad():
        for delta, vol, vb, ind, y in te_dl:
            logits = model(delta.to(dev), vol.to(dev), vb.to(dev), [t.to(dev) for t in ind])
            preds.extend(logits.argmax(-1).cpu().numpy())
            labels.extend(y.numpy())

    print("\n=== Multimodal Encoder Results ===")
    print(classification_report(labels, preds, target_names=["DOWN", "FLAT", "UP"]))
    print(confusion_matrix(labels, preds))

if __name__ == "__main__": main()
