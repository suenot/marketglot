"""Smoke test: extract every code cell from the generated .ipynb (except the
install cell), concatenate with small-size overrides, write to a temp .py
and run it in a subprocess with the project venv. Confirms end-to-end
pipeline on a tiny synthetic dataset."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

nb_path = Path("token_first_transformer/token_first_transformer.ipynb")
nb = json.loads(nb_path.read_text())

cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
assert len(cells) == 10, f"expected 10 code cells, got {len(cells)}"

source_blocks = ["".join(c["source"]) for c in cells[1:]]
# indices: 0=tokenizers, 1=dataset, 2=model, 3=trainer, 4=backtest,
# 5=config, 6=mock-data, 7=main, 8=save
assert "CFG = {" in source_blocks[5]
assert "Generate synthetic" in source_blocks[6]
assert "ARTIFACTS_ROOT" in source_blocks[8]

source_blocks[6] = source_blocks[6].replace("n_minutes=40000", "n_minutes=2000")
source_blocks[6] = source_blocks[6].replace(
    'months = ["2024-01", "2024-02", "2024-03", "2024-04"]',
    'months = ["2024-01", "2024-02", "2024-03"]',
)

override = '''
DATA_DIR = Path("/tmp/_colab_smoke_data")
CFG["data"]["train_months"] = ["2024-01", "2024-01"]
CFG["data"]["val_months"]   = ["2024-02", "2024-02"]
CFG["data"]["test_months"]  = ["2024-03", "2024-03"]
CFG["training"]["epochs"] = 1
CFG["training"]["batch_size"] = 16
CFG["training"]["checkpoint_dir"] = "/tmp/_colab_smoke_ckpt"
CFG["sequence"]["length"] = 32
CFG["sequence"]["target_horizon"] = 5
CFG["model"]["num_layers"] = 1
CFG["model"]["hidden_dim"] = 64
CFG["model"]["ffn_dim"] = 128
CFG["model"]["num_heads"] = 2
CFG["model"]["delta_emb_dim"] = 32
CFG["model"]["bucket_emb_dim"] = 8
ARTIFACTS_ROOT = Path("/tmp/_colab_smoke_artifacts")
'''

parts = source_blocks[:6] + [override] + source_blocks[6:]
# Collapse duplicate `from __future__ import annotations` — allowed only once
# at module top when cells are concatenated.
FUTURE = "from __future__ import annotations"
cleaned = [p.replace(FUTURE + "\n", "") for p in parts]
script = FUTURE + "\n\n" + "\n\n".join(cleaned)
tmp = Path("/tmp/_colab_smoke_script.py")
tmp.write_text(script)

venv_py = Path("token_first_transformer/.venv/bin/python")
cmd = [str(venv_py), str(tmp)]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print("--- STDOUT ---")
print(result.stdout)
print("--- STDERR ---")
print(result.stderr[-4000:])
print("exit code:", result.returncode)
sys.exit(result.returncode)
