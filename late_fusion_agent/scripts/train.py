"""Train full late-fusion pipeline."""
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
from dataset.fusion_dataset import FusionDataset
from training.fusion_trainer import FusionTrainer
from models.indicator_model import IndicatorModel


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
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    klines = data_dir / cfg["data"]["symbol"] / "klines_1m"
    s, e = cfg["data"]["train_months"]
    train_files = sorted([f for f in klines.glob("*.parquet") if s <= f.stem <= e])
    s, e = cfg["data"]["val_months"]
    val_files = sorted([f for f in klines.glob("*.parquet") if s <= f.stem <= e])
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    sc, tc = cfg["sequence"], cfg["tokenizer"]
    train_ds = FusionDataset(train_files, seq_len=sc["length"], target_horizon=sc["target_horizon"], target_threshold=sc["target_threshold"], range_pct=tc["delta"]["range_pct"], step_pct=tc["delta"]["step_pct"], n_bins=tc["bucket"]["n_bins"])
    val_ds = FusionDataset(val_files, seq_len=sc["length"], target_horizon=sc["target_horizon"], target_threshold=sc["target_threshold"], range_pct=tc["delta"]["range_pct"], step_pct=tc["delta"]["step_pct"], n_bins=tc["bucket"]["n_bins"])

    bs = cfg["training"]["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=0)

    ma = cfg["model_a"]
    model_a = PriceTransformer(delta_vocab_size=ma["delta_vocab_size"], bucket_vocab_size=ma["bucket_vocab_size"], delta_emb_dim=ma["delta_emb_dim"], bucket_emb_dim=ma["bucket_emb_dim"], hidden_dim=ma["hidden_dim"], num_layers=ma["num_layers"], num_heads=ma["num_heads"], ffn_dim=ma["ffn_dim"], dropout=ma["dropout"], num_classes=ma["num_classes"], seq_len=sc["length"])

    mb = cfg["model_b"]
    model_b = IndicatorModel(vocab_sizes=mb["vocab_sizes"], emb_dim=mb["emb_dim"], hidden_dim=mb["hidden_dim"], num_layers=mb["num_layers"], num_heads=mb["num_heads"], ffn_dim=mb["ffn_dim"], dropout=mb["dropout"], num_classes=mb["num_classes"], seq_len=sc["length"])

    tr = FusionTrainer(model_a, model_b, train_dl, val_dl, epochs_a=cfg["training"]["epochs_a"], epochs_b=cfg["training"]["epochs_b"], epochs_meta=cfg["training"]["epochs_meta"], lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"], early_stop_patience=cfg["training"]["early_stop_patience"], device=cfg["training"]["device"], checkpoint_dir=Path(cfg["training"]["checkpoint_dir"]))
    tr.train_all()


if __name__ == "__main__":
    main()
