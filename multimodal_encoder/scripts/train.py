"""Train multimodal encoder."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "token_first_transformer"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "indicator_tokenizer"))

import yaml, torch
from torch.utils.data import DataLoader
from dataset.multimodal_dataset import MultimodalDataset
from models.multimodal_model import MultimodalEncoder
from training.trainer import Trainer


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
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)

    kd = Path(cfg["data"]["data_dir"]) / cfg["data"]["symbol"] / "klines_1m"
    s,e = cfg["data"]["train_months"]
    tr_f = sorted([f for f in kd.glob("*.parquet") if s <= f.stem <= e])
    s,e = cfg["data"]["val_months"]
    va_f = sorted([f for f in kd.glob("*.parquet") if s <= f.stem <= e])
    print(f"Train: {len(tr_f)}, Val: {len(va_f)}")

    sc, tc = cfg["sequence"], cfg["tokenizer"]
    mk = dict(seq_len=sc["length"], target_horizon=sc["target_horizon"], target_threshold=sc["target_threshold"],
              range_pct=tc["delta"]["range_pct"], step_pct=tc["delta"]["step_pct"], n_bins=tc["bucket"]["n_bins"])
    tr_dl = DataLoader(MultimodalDataset(tr_f, **mk), batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate, num_workers=0)
    va_dl = DataLoader(MultimodalDataset(va_f, **mk), batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate, num_workers=0)

    mc, mi, mf = cfg["model"]["candle"], cfg["model"]["indicator"], cfg["model"]["fusion"]
    model = MultimodalEncoder(
        delta_vocab_size=mc["delta_vocab_size"], bucket_vocab_size=mc["bucket_vocab_size"],
        delta_emb_dim=mc["delta_emb_dim"], bucket_emb_dim=mc["bucket_emb_dim"], candle_proj_dim=mc["proj_dim"],
        ind_vocab_sizes=mi["vocab_sizes"], ind_emb_dim=mi["emb_dim"], ind_proj_dim=mi["proj_dim"],
        hidden_dim=mf["hidden_dim"], num_layers=mf["num_layers"], num_heads=mf["num_heads"],
        ffn_dim=mf["ffn_dim"], dropout=mf["dropout"], num_classes=mf["num_classes"], seq_len=sc["length"])
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    tr = Trainer(model, tr_dl, va_dl, epochs=cfg["training"]["epochs"], lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"], early_stop_patience=cfg["training"]["early_stop_patience"],
        device=cfg["training"]["device"], checkpoint_dir=Path(cfg["training"]["checkpoint_dir"]))
    tr.train()

if __name__ == "__main__": main()
