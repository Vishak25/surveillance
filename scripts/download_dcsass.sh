#!/usr/bin/env bash
set -euo pipefail

# DCSASS download helper using KaggleHub.
# Usage: ./scripts/download_dcsass.sh
#
# Prerequisites:
#   pip install kagglehub
#   Ensure your Kaggle credentials are configured via `kagglehub.login()` or environment variables.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/dcsass"
mkdir -p "$DATA_DIR"

# Show which Python is being used (helps with venv confusion)
echo "[INFO] Using python: $(command -v python3)"

# Simple import check for kagglehub
if ! python3 -c "import kagglehub" >/dev/null 2>&1; then
  echo "[ERROR] kagglehub not found. Install with 'pip install kagglehub'." >&2
  exit 1
fi

echo "[INFO] Downloading DCSASS dataset via KaggleHub..."
python3 - <<PY
import shutil
from pathlib import Path

import kagglehub

target_dir = Path("$DATA_DIR").resolve()
source_path = Path(kagglehub.dataset_download("mateohervas/dcsass-dataset")).resolve()
print(f"[INFO] Dataset cached at {source_path}")

for item in source_path.iterdir():
    destination = target_dir / item.name
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if item.is_dir():
        shutil.copytree(item, destination)
    else:
        shutil.copy2(item, destination)

print(f"[INFO] Dataset copied into {target_dir}")
PY

echo "[INFO] Download complete. Review README for expected layout."
