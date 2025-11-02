#!/usr/bin/env python3
"""Download DCSASS dataset from Kaggle using kagglehub."""

import kagglehub
import shutil
from pathlib import Path

# Download latest version
print("Downloading DCSASS dataset from Kaggle...")
path = kagglehub.dataset_download("mateohervas/dcsass-dataset")

print(f"Path to dataset files: {path}")

# Optionally, copy to your project's data directory
project_data_dir = Path(__file__).parent.parent / "surveillance_tf" / "data" / "dcsass"
project_data_dir.mkdir(parents=True, exist_ok=True)

print(f"\nDataset downloaded to: {path}")
print(f"You can copy it to your project at: {project_data_dir}")
print(f"\nTo copy the dataset, run:")
print(f"cp -r {path}/* {project_data_dir}/")

