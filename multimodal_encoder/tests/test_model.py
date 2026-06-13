import torch
import pytest
from models.multimodal_model import MultimodalEncoder

VS = [7, 9, 7, 8, 7, 7]

def test_output_shape():
    m = MultimodalEncoder(delta_vocab_size=122, bucket_vocab_size=10, delta_emb_dim=64, bucket_emb_dim=16,
        candle_proj_dim=128, ind_vocab_sizes=VS, ind_emb_dim=16, ind_proj_dim=128,
        hidden_dim=256, num_layers=2, num_heads=4, ffn_dim=512, dropout=0.1, num_classes=3, seq_len=32)
    B = 4
    logits = m(torch.randint(2,121,(B,32)), torch.randint(2,9,(B,32)), torch.randint(2,9,(B,32)),
               [torch.randint(2,vs-1,(B,32)) for vs in VS])
    assert logits.shape == (B, 3)

def test_single_input():
    m = MultimodalEncoder(delta_vocab_size=122, bucket_vocab_size=10, delta_emb_dim=16, bucket_emb_dim=8,
        candle_proj_dim=32, ind_vocab_sizes=VS, ind_emb_dim=8, ind_proj_dim=32,
        hidden_dim=64, num_layers=1, num_heads=2, ffn_dim=128, dropout=0.0, num_classes=3, seq_len=16)
    logits = m(torch.randint(2,121,(1,16)), torch.randint(2,9,(1,16)), torch.randint(2,9,(1,16)),
               [torch.randint(2,vs-1,(1,16)) for vs in VS])
    assert logits.shape == (1, 3)

def test_gradients():
    m = MultimodalEncoder(delta_vocab_size=122, bucket_vocab_size=10, delta_emb_dim=16, bucket_emb_dim=8,
        candle_proj_dim=32, ind_vocab_sizes=VS, ind_emb_dim=8, ind_proj_dim=32,
        hidden_dim=64, num_layers=1, num_heads=2, ffn_dim=128, dropout=0.0, num_classes=3, seq_len=16)
    m(torch.randint(2,121,(2,16)), torch.randint(2,9,(2,16)), torch.randint(2,9,(2,16)),
      [torch.randint(2,vs-1,(2,16)) for vs in VS]).sum().backward()
    for n, p in m.named_parameters():
        if p.requires_grad: assert p.grad is not None, f"No grad for {n}"

def test_finite():
    m = MultimodalEncoder(delta_vocab_size=122, bucket_vocab_size=10, delta_emb_dim=16, bucket_emb_dim=8,
        candle_proj_dim=32, ind_vocab_sizes=VS, ind_emb_dim=8, ind_proj_dim=32,
        hidden_dim=64, num_layers=1, num_heads=2, ffn_dim=128, dropout=0.0, num_classes=3, seq_len=16)
    m.eval()
    with torch.no_grad():
        logits = m(torch.randint(2,121,(1,16)), torch.randint(2,9,(1,16)), torch.randint(2,9,(1,16)),
                   [torch.randint(2,vs-1,(1,16)) for vs in VS])
    assert torch.isfinite(logits).all()
