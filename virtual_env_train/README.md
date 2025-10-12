# Remote Training Guide (DCSASS)

This folder contains everything you need to train the model in an isolated virtual environment (for example on a remote workstation or cloud VM) and then bring the resulting checkpoint back to this repository for evaluation and demo use.

## 1. Copy the training package
1. Copy the entire repository or, at minimum, the `surveillance_tf/` package directory and this `virtual_env_train/` folder to the training machine.
2. Ensure the DCSASS dataset is available on that machine (same layout as `data/dcsass/` here).

```
scp -r surveillance-tf <user>@<remote-host>:/path/to/workdir
```

## 2. Create & activate the training environment
```
cd /path/to/workdir/surveillance-tf
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r virtual_env_train/requirements_train.txt
pip install -e .
```

## 3. Prepare DCSASS splits (once per dataset copy)
If `data/dcsass/splits/*.csv` are missing, generate them:
```
python -m surveillance_tf.data.dcsass_loader \
  --data_root ./data/dcsass \
  --make_splits ./data/dcsass/splits
```

## 4. Run training
Use the helper script in this folder (wraps the main training module):
```
python virtual_env_train/train.py \
  --data_root ./data/dcsass \
  --train_csv ./data/dcsass/splits/train.csv \
  --val_csv   ./data/dcsass/splits/val.csv \
  --out ./outputs/dcsass_remote_run \
  --epochs 10
```

Key outputs:
- `./outputs/dcsass_remote_run/` → TensorBoard logs + intermediate checkpoints.
- `./models/movinet/ckpt_best/` → SavedModel of the best checkpoint (ready for eval/demo).

## 5. Bring results back
Copy the following artefacts to your local repo:
- `outputs/dcsass_remote_run/` (or whichever run directory you specified).
- `models/movinet/ckpt_best/` (overwrite or version as needed).

```
scp -r <user>@<remote-host>:/path/to/workdir/surveillance-tf/models/movinet/ckpt_best ./models/movinet/
scp -r <user>@<remote-host>:/path/to/workdir/surveillance-tf/outputs/dcsass_remote_run ./outputs/
```

## 6. Evaluate / Demo locally
Back on your local machine (using the main repo):
```
python -m surveillance_tf.train.eval_mil_ucfcrime \
  --data_root ./data/dcsass \
  --test_csv ./data/dcsass/splits/test.csv \
  --ckpt ./models/movinet/ckpt_best \
  --out ./outputs/dcsass_remote_run_eval

python -m surveillance_tf.demo.coord \
  --data_root ./data/dcsass \
  --video "./data/dcsass/sample/*.mp4" \
  --ckpt ./models/movinet/ckpt_best \
  --config ./configs/thresholds.yaml --fps 25
```

## Notes
- The training script exposes all hyper-parameters (`--epochs`, `--lr`, `--T`, `--stride`, etc.) identical to the in-repo CLI.
- Environment variables that mitigate macOS/ffmpeg mutex issues are set in the main training module; no extra configuration is required beyond the commands above.
- If you need to change the dataset location, update `--data_root` or set the `DCSASS_DATA_DIR` environment variable before running scripts.
