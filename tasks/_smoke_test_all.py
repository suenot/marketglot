"""Smoke-test every generated <project>.ipynb end-to-end.

For each project:
  1. Read the .ipynb.
  2. Concatenate every code cell (skipping the `!pip install` cell).
  3. Apply tiny overrides (fewer months, smaller model, 1 epoch).
  4. Execute with the project venv in a subprocess.
  5. Assert exit code 0.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "token_first_transformer" / ".venv" / "bin" / "python"


def load_code_cells(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text())
    blocks: list[str] = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith("!"):
            continue
        blocks.append(src)
    return blocks


def collapse_futures(parts: list[str]) -> str:
    FUTURE = "from __future__ import annotations"
    cleaned = [p.replace(FUTURE + "\n", "") for p in parts]
    return FUTURE + "\n\n" + "\n\n".join(cleaned)


def run(tag: str, script: str, tmp: Path) -> None:
    tmp.write_text(script)
    print(f"\n===== {tag} =====")
    r = subprocess.run([str(VENV), str(tmp)], capture_output=True, text=True, timeout=600)
    print("--- stdout (tail) ---")
    print("\n".join(r.stdout.splitlines()[-30:]))
    if r.returncode != 0:
        print("--- stderr (tail) ---")
        print("\n".join(r.stderr.splitlines()[-50:]))
        print(f"[FAIL] {tag} exit={r.returncode}")
        sys.exit(r.returncode)
    print(f"[OK] {tag}")


def smoke_indicator_tokenizer() -> None:
    nb = ROOT / "indicator_tokenizer" / "indicator_tokenizer.ipynb"
    parts = load_code_cells(nb)
    override = '''
DATA_DIR = Path("/tmp/_smoke_ind_data")
MOCK_MINUTES_PER_MONTH = 2000
MOCK_MONTHS = ["2024-01", "2024-02"]
TRAIN_MONTHS = ("2024-01", "2024-02")
BOUNDARIES_DIR = Path("/tmp/_smoke_ind_boundaries")
ARTIFACTS_ROOT = Path("/tmp/_smoke_ind_artifacts")
'''
    # Insert override right after the config block. Simpler: append after all
    # Path-related blocks by prepending to the config. We inject the override
    # between blocks[2] (config cell) and blocks[3] (mock-data). Cells after
    # dropping install: 0=IndicatorComputer, 1=IndicatorTokenizer, 2=config,
    # 3=mock data, 4=fit, 5=verify.
    assert "BOUNDARIES_DIR" in parts[2]
    parts = parts[:3] + [override] + parts[3:]
    script = collapse_futures(parts)
    run("indicator_tokenizer", script, Path("/tmp/_smoke_ind.py"))


def smoke_late_fusion() -> None:
    nb = ROOT / "late_fusion_agent" / "late_fusion_agent.ipynb"
    parts = load_code_cells(nb)
    # After dropping install: 0 tokenizers, 1 comp, 2 ind_tok, 3 price_tr,
    # 4 ind_model, 5 meta, 6 dataset, 7 trainer, 8 config, 9 mock_data,
    # 10 make_split, 11 main
    assert "CFG = {" in parts[8]
    override = '''
DATA_DIR = Path("/tmp/_smoke_lfa_data")
MOCK_MINUTES_PER_MONTH = 2500
MOCK_MONTHS = ["2024-01", "2024-02", "2024-03"]
CFG["data"]["train_months"] = ["2024-01", "2024-01"]
CFG["data"]["val_months"]   = ["2024-02", "2024-02"]
CFG["data"]["test_months"]  = ["2024-03", "2024-03"]
CFG["sequence"]["length"] = 32
CFG["sequence"]["target_horizon"] = 5
CFG["training"]["batch_size"] = 16
CFG["training"]["epochs_a"] = 1
CFG["training"]["epochs_b"] = 1
CFG["training"]["epochs_meta"] = 2
CFG["training"]["checkpoint_dir"] = "/tmp/_smoke_lfa_ckpt"
CFG["model_a"]["hidden_dim"] = 64
CFG["model_a"]["num_layers"] = 1
CFG["model_a"]["ffn_dim"] = 128
CFG["model_a"]["num_heads"] = 2
CFG["model_a"]["delta_emb_dim"] = 32
CFG["model_a"]["bucket_emb_dim"] = 8
CFG["model_b"]["hidden_dim"] = 64
CFG["model_b"]["num_layers"] = 1
CFG["model_b"]["ffn_dim"] = 128
CFG["model_b"]["num_heads"] = 2
CFG["model_b"]["emb_dim"] = 8
ARTIFACTS_ROOT = Path("/tmp/_smoke_lfa_artifacts")
'''
    parts = parts[:9] + [override] + parts[9:]
    script = collapse_futures(parts)
    run("late_fusion_agent", script, Path("/tmp/_smoke_lfa.py"))


def smoke_multimodal() -> None:
    nb = ROOT / "multimodal_encoder" / "multimodal_encoder.ipynb"
    parts = load_code_cells(nb)
    # After install: 0 tok, 1 comp, 2 ind_tok, 3 model, 4 dataset, 5 trainer,
    # 6 config, 7 mock, 8 make_split, 9 main
    assert "CFG = {" in parts[6]
    override = '''
DATA_DIR = Path("/tmp/_smoke_mm_data")
MOCK_MINUTES_PER_MONTH = 2500
MOCK_MONTHS = ["2024-01", "2024-02", "2024-03"]
CFG["data"]["train_months"] = ["2024-01", "2024-01"]
CFG["data"]["val_months"]   = ["2024-02", "2024-02"]
CFG["data"]["test_months"]  = ["2024-03", "2024-03"]
CFG["sequence"]["length"] = 32
CFG["sequence"]["target_horizon"] = 5
CFG["training"]["batch_size"] = 16
CFG["training"]["epochs"] = 1
CFG["training"]["checkpoint_dir"] = "/tmp/_smoke_mm_ckpt"
CFG["model"]["candle"]["proj_dim"] = 32
CFG["model"]["candle"]["delta_emb_dim"] = 32
CFG["model"]["candle"]["bucket_emb_dim"] = 8
CFG["model"]["indicator"]["proj_dim"] = 32
CFG["model"]["indicator"]["emb_dim"] = 8
CFG["model"]["fusion"]["hidden_dim"] = 64
CFG["model"]["fusion"]["num_layers"] = 1
CFG["model"]["fusion"]["ffn_dim"] = 128
CFG["model"]["fusion"]["num_heads"] = 2
ARTIFACTS_ROOT = Path("/tmp/_smoke_mm_artifacts")
'''
    parts = parts[:7] + [override] + parts[7:]
    script = collapse_futures(parts)
    run("multimodal_encoder", script, Path("/tmp/_smoke_mm.py"))


def smoke_moe() -> None:
    nb = ROOT / "moe_trading_agent" / "moe_trading_agent.ipynb"
    parts = load_code_cells(nb)
    # After install: 0 tok, 1 comp, 2 ind_tok, 3 router, 4 moe_model,
    # 5 dataset, 6 trainer, 7 config, 8 mock, 9 main
    assert "MODEL_CFG" in parts[7]
    override = '''
DATA_DIR = Path("/tmp/_smoke_moe_data")
MOCK_MINUTES_PER_MONTH = 3000
MOCK_MONTHS = ["2024-01", "2024-02"]
MODEL_CFG["seq_len"] = 32
MODEL_CFG["num_experts"] = 4
MODEL_CFG["top_k"] = 2
MODEL_CFG["num_layers"] = 1
MODEL_CFG["num_heads"] = 2
MODEL_CFG["dim"] = 256  # keep 256: model hard-codes projections to 128+128=256
MODEL_CFG["hidden_dim"] = 128
TRAIN_CFG["batch_size"] = 16
TRAIN_CFG["epochs"] = 1
TRAIN_CFG["horizon"] = 5
TRAIN_CFG["checkpoint_dir"] = "/tmp/_smoke_moe_ckpt"
ARTIFACTS_ROOT = Path("/tmp/_smoke_moe_artifacts")
'''
    parts = parts[:8] + [override] + parts[8:]
    script = collapse_futures(parts)
    run("moe_trading_agent", script, Path("/tmp/_smoke_moe.py"))


if __name__ == "__main__":
    import shutil
    for d in ["/tmp/_smoke_ind_data", "/tmp/_smoke_ind_boundaries", "/tmp/_smoke_ind_artifacts",
              "/tmp/_smoke_lfa_data", "/tmp/_smoke_lfa_ckpt", "/tmp/_smoke_lfa_artifacts",
              "/tmp/_smoke_mm_data", "/tmp/_smoke_mm_ckpt", "/tmp/_smoke_mm_artifacts",
              "/tmp/_smoke_moe_data", "/tmp/_smoke_moe_ckpt", "/tmp/_smoke_moe_artifacts"]:
        shutil.rmtree(d, ignore_errors=True)

    smoke_indicator_tokenizer()
    smoke_late_fusion()
    smoke_multimodal()
    smoke_moe()
    print("\nALL SMOKE TESTS PASSED")
