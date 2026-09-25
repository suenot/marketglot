import random

import numpy as np
import torch
import pytest
from training.trainer import Trainer, ResumableRandomSampler, compute_class_weights
from models.price_transformer import PriceTransformer
from scripts.train import seed_all


def _make_model():
    return PriceTransformer(
        delta_vocab_size=122, bucket_vocab_size=10,
        delta_emb_dim=16, bucket_emb_dim=8,
        hidden_dim=32, num_layers=1, num_heads=2,
        ffn_dim=64, dropout=0.0, num_classes=3, seq_len=16,
    )


def _make_dataloader(n=32, seq_len=16, shuffle=False, resumable=False):
    delta = torch.randint(2, 121, (n, seq_len))
    vol = torch.randint(2, 9, (n, seq_len))
    vb = torch.randint(2, 9, (n, seq_len))
    labels = torch.randint(0, 3, (n,))
    dataset = list(zip(delta, vol, vb, labels))
    return torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=shuffle and not resumable,
        sampler=ResumableRandomSampler(dataset, seed=7) if resumable else None,
        generator=torch.Generator() if shuffle or resumable else None,
    )


def test_seed_all_repeats_model_initialization_and_rng_streams():
    seed_all(7)
    first_model = _make_model()
    first_draws = (random.random(), np.random.random(), torch.rand(1))

    seed_all(7)
    second_model = _make_model()
    second_draws = (random.random(), np.random.random(), torch.rand(1))

    for key, value in first_model.state_dict().items():
        torch.testing.assert_close(value, second_model.state_dict()[key], rtol=0, atol=0)
    assert first_draws[0] == second_draws[0]
    assert first_draws[1] == second_draws[1]
    torch.testing.assert_close(first_draws[2], second_draws[2], rtol=0, atol=0)


def test_resumable_sampler_skips_completed_samples():
    sampler = ResumableRandomSampler(range(20), seed=7)
    sampler.set_epoch(2)
    full = list(sampler)
    sampler.set_epoch(2, start_index=8)
    assert list(sampler) == full[8:]


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
    trainer._val_epoch = lambda: (1.0, 0.5)
    metrics = trainer.train()
    assert len(metrics) == 3


def test_mid_epoch_resume_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(123)
    original = _make_model()
    initial = {key: value.clone() for key, value in original.state_dict().items()}
    train_dl = _make_dataloader(n=24, resumable=True)
    val_dl = _make_dataloader(n=8)

    uninterrupted = Trainer(original, train_dl, val_dl, epochs=1, device="cpu",
                            grad_accum_steps=2, checkpoint_dir=tmp_path / "full",
                            checkpoint_every_steps=1, seed=7, max_threads=1)
    full_metrics = uninterrupted.train()
    final = {key: value.clone() for key, value in original.state_dict().items()}

    interrupted_model = _make_model()
    interrupted_model.load_state_dict(initial)
    interrupted = Trainer(interrupted_model, train_dl, val_dl, epochs=1, device="cpu",
                          grad_accum_steps=2, checkpoint_dir=tmp_path / "partial",
                          checkpoint_every_steps=1, seed=7, max_threads=1)
    save = interrupted._save_checkpoint

    def stop_after_first_step(name, epoch):
        save(name, epoch)
        if name == "latest.pt" and interrupted.next_batch == 2:
            raise RuntimeError("simulated shutdown")

    interrupted._save_checkpoint = stop_after_first_step
    with pytest.raises(RuntimeError, match="simulated shutdown"):
        interrupted.train()

    resumed_model = _make_model()
    resumed = Trainer(resumed_model, train_dl, val_dl, epochs=1, device="cpu",
                      grad_accum_steps=2, checkpoint_dir=tmp_path / "resumed",
                      checkpoint_every_steps=1, seed=7, max_threads=1,
                      resume_from=tmp_path / "partial" / "latest.pt")
    assert resumed.next_batch == 2
    assert resumed.train() == full_metrics
    for key, expected in final.items():
        torch.testing.assert_close(resumed_model.state_dict()[key], expected, rtol=0, atol=0)
