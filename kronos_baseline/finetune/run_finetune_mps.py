"""Finetune Kronos on Apple GPU (MPS).

Kronos's own finetune pipeline only selects CUDA-or-CPU. On a Mac that means CPU,
which is pointless when an Apple GPU is available. This launcher reuses Kronos's
`SequentialTrainer` unchanged but **forces the device to MPS** (its train loops
take `device` as an argument and have no CUDA-only ops — no AMP/GradScaler/.cuda()).

It generates a config on the fly, downloads the pretrained tokenizer to a local
dir (so the predictor-only phase can find it), and runs the basemodel (predictor)
finetune by default (tokenizer finetune is opt-in; its BSQ entropy ops are riskier
on MPS).

Example:
    KRONOS_PATH=../Kronos python kronos_baseline/finetune/run_finetune_mps.py \
        --csv kronos_baseline/finetune/data/BTCUSDT_1m.csv \
        --out-dir kronos_baseline/finetune/runs/btc \
        --epochs 1 --batch-size 16 --lookback 256 --predict 60
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# MPS: let unsupported ops fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import yaml


def resolve_kronos() -> Path:
    cand = os.environ.get("KRONOS_PATH") or (Path(__file__).resolve().parents[3] / "Kronos")
    p = Path(cand).expanduser().resolve()
    if not (p / "finetune_csv" / "train_sequential.py").is_file():
        raise FileNotFoundError(f"Kronos finetune_csv not found at {p}; set KRONOS_PATH")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Finetune Kronos on MPS (Apple GPU)")
    ap.add_argument("--csv", required=True, help="training CSV (see prepare_csv.py)")
    ap.add_argument("--out-dir", required=True, help="where finetuned models/logs go")
    ap.add_argument("--pretrained-predictor", default="NeoQuasar/Kronos-small")
    ap.add_argument("--pretrained-tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lookback", type=int, default=256)
    ap.add_argument("--predict", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-5, help="predictor learning rate")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--finetune-tokenizer", action="store_true",
                    help="also finetune the tokenizer (riskier on MPS); default off")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing checkpoint in --out-dir (start from the pretrained model)")
    ap.add_argument("--device", default=None, help="override device (mps/cpu/cuda)")
    a = ap.parse_args()

    kronos = resolve_kronos()
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) resolve pretrained weights to local cache dirs (reuse HF cache; no re-download).
    #    The predictor phase needs a real on-disk tokenizer dir (it does os.path.exists).
    from huggingface_hub import snapshot_download

    def resolve_repo(repo: str) -> str:
        if Path(repo).expanduser().is_dir():
            return str(Path(repo).expanduser().resolve())
        try:
            return snapshot_download(repo_id=repo)            # uses cache, downloads only if missing
        except Exception:
            return snapshot_download(repo_id=repo, local_files_only=True)  # offline: cache only

    tok_dir = resolve_repo(a.pretrained_tokenizer)
    # auto-resume: continue from a previously saved checkpoint in this out-dir, so
    # repeated short (~30 min) runs accumulate instead of restarting from scratch.
    resume_ckpt = out_dir / "mps_finetune" / "basemodel" / "best_model"
    if (not a.no_resume) and (resume_ckpt / "config.json").exists():
        pred_dir = str(resume_ckpt)
        print(f"RESUMING predictor from existing checkpoint: {pred_dir}")
    else:
        pred_dir = resolve_repo(a.pretrained_predictor)
    print(f"tokenizer: {tok_dir}\npredictor: {pred_dir}")

    # 2) build config
    cfg = {
        "data": {
            "data_path": str(Path(a.csv).resolve()),
            "lookback_window": a.lookback, "predict_window": a.predict,
            "max_context": max(512, a.lookback), "clip": 5.0,
            "train_ratio": 0.9, "val_ratio": 0.1, "test_ratio": 0.0,
        },
        "training": {
            "tokenizer_epochs": a.epochs, "basemodel_epochs": a.epochs,
            "batch_size": a.batch_size, "log_interval": 10, "num_workers": a.num_workers,
            "seed": 42, "tokenizer_learning_rate": 2e-4, "predictor_learning_rate": a.lr,
            "adam_beta1": 0.9, "adam_beta2": 0.95, "adam_weight_decay": 0.1,
            "accumulation_steps": 1,
        },
        "model_paths": {
            "pretrained_tokenizer": tok_dir,
            "pretrained_predictor": pred_dir,
            "exp_name": "mps_finetune", "base_path": str(out_dir),
            "base_save_path": "", "finetuned_tokenizer": tok_dir,
            "tokenizer_save_name": "tokenizer", "basemodel_save_name": "basemodel",
        },
        "experiment": {
            "name": "kronos_mps_finetune", "description": "MPS finetune", "use_comet": False,
            "train_tokenizer": bool(a.finetune_tokenizer), "train_basemodel": True,
            "skip_existing": False, "pre_trained_tokenizer": True, "pre_trained_predictor": True,
        },
        "device": {"use_cuda": False, "device_id": 0},
        "distributed": {"use_ddp": False, "backend": "gloo"},
    }
    cfg_path = out_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"config -> {cfg_path}")

    # 3) run Kronos' SequentialTrainer with the device forced to MPS
    sys.path.insert(0, str(kronos))
    sys.path.insert(0, str(kronos / "finetune_csv"))
    os.chdir(kronos / "finetune_csv")  # its imports use relative '../'
    from train_sequential import SequentialTrainer  # type: ignore

    if a.device:
        dev = torch.device(a.device)
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")

    trainer = SequentialTrainer(str(cfg_path))
    trainer.device = dev  # <-- the whole point: force Apple GPU
    print(f">>> forced training device: {trainer.device}")
    ok = trainer.run_training()
    print("FINETUNE OK" if ok else "FINETUNE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
