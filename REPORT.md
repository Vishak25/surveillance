# 1) Executive Summary (bullets)
- **Purpose**: TensorFlow 2.x reproduction of the MIL anomaly detector from Sultani et al. with a DCSASS-focused runtime demo (Streamlit UI, incident logging, PDF reports).
- **Datasets expected**: Only DCSASS is required (root `./data/dcsass`); UCF-Crime support has been removed.
- **Current layout readiness**: With `data/dcsass/{DCSASS Dataset/, sample/, metadata.csv}`, all pipelines (splits, training, evaluation, demo) run without further edits.

# 2) Project Map
```
surveillance_tf/
├── data/
│   ├── dcsass_loader.py
│   └── transforms.py
├── demo/
│   ├── coord.py
│   ├── incidents.py
│   ├── responder.py
│   └── sort_tracker.py
├── export/save_models.py
├── losses/mil_losses.py
├── models/
│   ├── mil_head.py
│   └── movinet_backbone.py
├── train/
│   ├── train_mil_ucfcrime.py  (DCSASS trainer)
│   └── eval_mil_ucfcrime.py   (DCSASS evaluator)
├── utils/
│   ├── logging.py
│   ├── metrics.py
│   ├── paths.py
│   ├── ring_buffer.py
│   └── seed.py
└── __init__.py
```
- `data/dcsass_loader.py`: Metadata-aware loader producing ragged segment batches; writes split CSVs when `metadata.csv` absent.
- `data/transforms.py`: Video decoding (imageio-ffmpeg) and segment creation utilities.
- `utils/paths.py`: Discovers the DCSASS dataset root (`resolve_dcsass_root`), lists videos/directories.
- `train/train_mil_ucfcrime.py`: DCSASS training script (historical filename). Builds MoViNet backbone + MIL head, optimises ranking/sparsity/smoothness losses, saves checkpoints.
- `train/eval_mil_ucfcrime.py`: Evaluates SavedModel checkpoints, produces ROC/AUC metrics.
- `demo/coord.py`: CLI incident demo performing sliding-window inference, incident tracking, clip/report export.
- `demo/incidents.py`: SQLite DAO schema for incident persistence.
- `demo/responder.py`: Evidence clip export (cv2), report rendering (Jinja2 + WeasyPrint), optional Discord webhook.
- `export/save_models.py`: Exports backbone/head SavedModels.
- `losses/mil_losses.py`: MIL ranking/sparsity/smoothness loss implementations.
- `models/`: MoViNet backbone wrapper and MIL scoring head.
- `utils/`: Logging, metrics (ROC/PR, TTFA), deterministic seeding, ring buffer.

**Command-line entry points**
| Module | Purpose | Key args | Example |
| --- | --- | --- | --- |
| `python -m surveillance_tf.data.dcsass_loader` | Inspect/split DCSASS | `--data_root`, `--split`, `--csv`, `--make_splits`, `--sample` | `python -m surveillance_tf.data.dcsass_loader --data_root ./data/dcsass --make_splits ./data/dcsass/splits` |
| `python -m surveillance_tf.train.train_mil_ucfcrime` | Train MIL model | `--config`, `--out`, `--epochs`, `--lr`, `--mil_mode`, `--T`, `--stride`, `--image_size` | `python -m surveillance_tf.train.train_mil_ucfcrime --config ./configs/experiments/oneclass_dcsass.yaml --out ./outputs/dcsass_run1` |
| `python -m surveillance_tf.train.eval_mil_ucfcrime` | Evaluate MIL model | `--data_root`, `--test_csv`, `--ckpt`, `--out` | `python -m surveillance_tf.train.eval_mil_ucfcrime --data_root ./data/dcsass --test_csv ./data/dcsass/splits/test.csv --ckpt ./models/movinet/ckpt_best --out ./outputs/dcsass_run1` |
| `python -m surveillance_tf.demo.coord` | CLI incident demo | `--data_root`, `--video`, `--ckpt`, `--config`, `--fps`, `--db`, `--out`, `--passes` | `python -m surveillance_tf.demo.coord --data_root ./data/dcsass --video "./data/dcsass/sample/*.mp4" --ckpt ./models/movinet/ckpt_best --config ./configs/thresholds.yaml --fps 25` |
| `python -m surveillance_tf.export.save_models` | Export backbone/head | `--output-dir`, `--window`, `--head-units` | `python -m surveillance_tf.export.save_models --output-dir ./models/export` |
| `python -m streamlit run surveillance_tf/demo/ui_app.py` | Streamlit UI | (Streamlit args) | `python -m streamlit run surveillance_tf/demo/ui_app.py` |

# 3) Data & Paths
- **Root resolution** (`utils/paths.py`)
  - `resolve_dcsass_root(explicit_root)` checks explicit CLI path, `DCSASS_DATA_DIR`, repo default `data/dcsass`, then KaggleHub cache. Errors list all attempted paths and expected tree.
  - `list_videos(root, exts)`: recursive finder for `.mp4/.avi/.mov/.mkv`.
  - `find_video_dirs(root)`: prioritises `DCSASS Dataset/` and `sample/`, otherwise scans full root.
- **Loader** (`data/dcsass_loader.py`)
  - Reads `metadata.csv` if present (auto-detects path/label/split columns; resolves relative paths). Adds stratified splits if missing.
  - Without metadata, scans recursively, infers labels, creates 80/10/10 stratified splits, writes `data/dcsass/splits/{train,val,test}.csv`.
  - `load_split_entries(root, split, seed, csv_path)` returns entries with `path`, `label`, `label_index`, and `binary_label` (normal=0, abnormal=1).
  - `make_bag_dataset(root, split, T, stride, batch_size, image_size, csv_path)` yields ragged tensors `[num_segments, T, H, W, 3]`, binary label, and video ID.

# 4) Model & Training
- **Backbone**: MoViNet-A0 (TF-Hub) via `models/movinet_backbone.py`, input `(None, H, W, 3)` with dropout option.
- **Segments**: `T` frames (default 32), stride 3, resized to `image_size` (default 224×224).
- **Head**: `models/mil_head.MILScoringHead` (Dense `[256,128]`, dropout 0.2, sigmoid output).
- **Losses** (`losses/mil_losses.py`): ranking hinge, sparsity (mean positive scores), smoothness (adjacent diff squared) combined as `ranking + λ_sparse*sparsity + λ_smooth*smoothness` (defaults λ_sparse=8e-5, λ_smooth=0.1, margin 1.0).
- **Training loop** (`train/train_mil_ucfcrime.py`):
  - Loads DCSASS splits (defaults to `data/dcsass/splits/{train,val}.csv` if not provided).
  - Builds encoder/head, Adam optimiser.
  - Constructs positive/negative datasets from ragged batches, iterates `min(#pos,#neg)` steps.
  - Logs to TensorBoard (`outputs/<run>/logs/`), saves best SavedModel in `outputs/<run>/checkpoints/ckpt_best` and copies to `models/movinet/ckpt_best`; writes `training_state.json` with epoch/AUC.

# 5) Evaluation
- **Script**: `train/eval_mil_ucfcrime.py` (DCSASS-only). Loads test dataset via `make_bag_dataset` (dense via `.to_tensor()`), runs SavedModel, aggregates scores by `max` segment per video.
- **Outputs**: `roc_curve.png`, `scores.csv` (label, score), AUC logged.

# 6) Demo / Runtime
- **CLI**: `demo/coord.py` with args `--data_root`, `--video`, `--ckpt`, `--config`, `--fps`, `--db`, `--out`, `--passes`.
- **Inference**: Sliding window (length `window_frames`, stride `stride_frames` from config), per-window MC-dropout `--passes` to estimate mean/std; `fuse_scores` currently returns anomaly mean.
- **Config**: `configs/thresholds.yaml` defines thresholds, CI floor, cooldown seconds, window/stride, severity class lists used in FSM.
- **Incidents**: `IncidentCoordinator` merges within cooldown, transitions OPEN→READY→ALERTED. Logged to SQLite via `demo/incidents.IncidentDAO` (schema includes severity, class, scores, clip/report paths).
- **Evidence/reporting**: Clips produced via `ring_buffer.export_clip` (cv2 writer). Snapshots saved with cv2. Reports rendered using `demo/responder.render_report` (Jinja2 template + WeasyPrint). `maybe_notify_discord` posts to webhook if configured.
- **UI**: `demo/ui_app.py` leverages DAO/coordinator for Streamlit dashboard, threshold knobs, report downloads.

# 7) Multiprocessing / macOS Safety
- No multiprocessing or process pools. Heavy imports happen after environment variables clamp threading (`OPENCV_OPENCL_RUNTIME`, `TF_NUM_*`, `IMAGEIO_FFMPEG_THREADS`).
- Potential mutex hotspots (only if future multiprocessing introduced): the training/eval loops convert ragged tensors to dense via `.to_tensor()`; ensure `spawn` start method is set before TensorFlow import should multiprocessing ever be added.

# 8) DCSASS Readiness Checklist
| Item | Status | Reference |
| --- | --- | --- |
| Autodetect `./data/dcsass` when `--data_root` missing | ✅ | `utils/paths.resolve_dcsass_root` |
| All CLIs accept `--data_root` (DCSASS-only) | ✅ | `data/dcsass_loader.py`, `train/train_mil_ucfcrime.py`, `train/eval_mil_ucfcrime.py`, `demo/coord.py` |
| Use `metadata.csv` if present; otherwise auto-create splits | ✅ | `data/dcsass_loader.py::_load_metadata` |
| Works with videos under `DCSASS Dataset/` & `sample/` | ✅ | `utils/paths.find_video_dirs`, `list_videos` |
| No hard-coded `ucf_crime` references remain | ✅ | Repository now DCSASS-only |

# 9) Actionable Fixes (ordered, copy-pasteable)
✅ Centralise dataset resolution (`resolve_dcsass_root`).
✅ Update all CLIs to accept only `--data_root` (DCSASS).
✅ Remove UCF-Crime code/docs/tests.
✅ Ensure loader handles metadata + auto-splits.
✅ Refresh README/Makefile examples for DCSASS-only workflow.

# 10) Quick Smoke Plan
- **Without dataset** (expect friendly messages):
  - `python -m surveillance_tf.data.dcsass_loader --data_root ./data/dcsass --split train --sample`
  - `python -m surveillance_tf.train.train_mil_ucfcrime --data_root ./data/dcsass --out ./outputs/dry_run`
  - `python -m surveillance_tf.demo.coord --data_root ./data/dcsass --video "./data/dcsass/sample/*.mp4" --ckpt ./models/movinet/ckpt_best --config ./configs/thresholds.yaml --fps 25`
- **With dataset present**:
  1. `python -m surveillance_tf.data.dcsass_loader --data_root ./data/dcsass --make_splits ./data/dcsass/splits`
  2. `python -m surveillance_tf.train.train_mil_ucfcrime --config ./configs/experiments/oneclass_dcsass.yaml --out ./outputs/dcsass_run1 --epochs 1`
  3. `python -m surveillance_tf.train.eval_mil_ucfcrime --data_root ./data/dcsass --test_csv ./data/dcsass/splits/test.csv --ckpt ./models/movinet/ckpt_best --out ./outputs/dcsass_run1`
  4. `python -m surveillance_tf.demo.coord --data_root ./data/dcsass --video "./data/dcsass/sample/*.mp4" --ckpt ./models/movinet/ckpt_best --config ./configs/thresholds.yaml --fps 25`

# 11) Dependency/Version Notes
- Requirements specify minimum versions: TensorFlow ≥ 2.12, TensorFlow Hub, OpenCV, imageio/imageio-ffmpeg, NumPy, SciKit-learn, pandas, WeasyPrint, etc.
- macOS/Apple Silicon: ensure compatible TensorFlow build; environment variables already set to disable OpenCL and limit threads to avoid mutex crashes.

# 12) Optional UCF-Crime
Removed—repository is now DCSASS-only.
