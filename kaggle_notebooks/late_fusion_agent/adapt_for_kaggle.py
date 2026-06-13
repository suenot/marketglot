import json
from pathlib import Path

nb = json.loads(Path("late_fusion_agent.ipynb").read_text())

changes = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    new_src = src
    # Replace paths
    new_src = new_src.replace("/content/", "/kaggle/working/")
    # Remove google.colab imports
    if "from google.colab import drive" in new_src:
        lines = new_src.splitlines()
        filtered = []
        skip = False
        for line in lines:
            if "from google.colab import drive" in line:
                skip = True
                continue
            if skip and line.strip().startswith("drive.mount"):
                skip = False
                continue
            filtered.append(line)
        new_src = "\n".join(filtered)
    if new_src != src:
        cell["source"] = [new_src] if "\n" not in new_src else [line + "\n" for line in new_src.splitlines()]
        # Remove trailing newline on last element if present
        if cell["source"] and cell["source"][-1].endswith("\n"):
            cell["source"][-1] = cell["source"][-1][:-1]
        changes += 1

# Update markdown title
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown" and "late_fusion_agent" in "".join(cell["source"]):
        src = "".join(cell["source"])
        if "Colab" in src:
            new_src = src.replace("Colab", "Kaggle")
            cell["source"] = [new_src]
            changes += 1
        break

Path("late_fusion_agent.ipynb").write_text(json.dumps(nb, indent=1))
print(f"adapted {changes} cells")
