import torch
import pytest
from training.trainer import Trainer, compute_class_weights
from models.price_transformer import PriceTransformer


def _make_model():
    return PriceTransformer(
        delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=16, bucket_emb_dim=8,
        hidden_dim=32, num_layers=1, num_heads=2,
        ffn_dim=64, dropout=0.0, num_classes=3, seq_len=16,
    )


def _make_dataloader(n=32, seq_len=16):
    delta = torch.randint(2, 121, (n, seq_len))
    vol = torch.randint(2, 9, (n, seq_len))
    vb = torch.randint(2, 9, (n, seq_len))
    labels = torch.randint(0, 3, (n,))
    return torch.utils.data.DataLoader(
        list(zip(delta, vol, vb, labels)), batch_size=8,
    )


def test_compute_class_weights():
    labels = [0, 0, 1, 1, 1, 1, 2, 2]
    weights = compute_class_weights(labels, num_classes=3)
    assert len(weights) == 3
    assert weights[1] < weights[0]
    assert weights[1] < weights[2]


def test_trainer_one_epoch(tmp_path):
    model = _make_model()
    train_dl = _make_dataloader()
    val_dl = _make_dataloader(n=16)
    trainer = Trainer(
        model=model, train_loader=train_dl, val_loader=val_dl,
        epochs=1, lr=1e-3, device="cpu", checkpoint_dir=tmp_path,
    )
    metrics = trainer.train()
    assert len(metrics) == 1
    assert "train_loss" in metrics[0]
    assert "val_loss" in metrics[0]
    assert "val_f1" in metrics[0]


def test_trainer_saves_checkpoint(tmp_path):
    model = _make_model()
    train_dl = _make_dataloader()
    val_dl = _make_dataloader(n=8)
    trainer = Trainer(
        model=model, train_loader=train_dl, val_loader=val_dl,
        epochs=1, lr=1e-3, device="cpu", checkpoint_dir=tmp_path,
    )
    trainer.train()
    ckpts = list(tmp_path.glob("*.pt"))
    assert len(ckpts) >= 1


def test_trainer_early_stop(tmp_path):
    model = _make_model()
    train_dl = _make_dataloader()
    val_dl = _make_dataloader(n=8)
    trainer = Trainer(
        model=model, train_loader=train_dl, val_loader=val_dl,
        epochs=50, lr=1e-3, device="cpu", checkpoint_dir=tmp_path,
        early_stop_patience=2,
    )
    metrics = trainer.train()
    assert len(metrics) <= 5
