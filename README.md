# Surveillance-TF

Reproducible TensorFlow 2.x implementation of the multiple-instance learning anomaly detector from Sultani et al. (CVPR 2018), extended with a DCSASS-based runtime demo that performs sliding-window inference, incident management, evidence clipping, and PDF reporting via Streamlit.

## Quickstart
1. **Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```
2. **Datasets**
   - **DCSASS**: authenticate with Kaggle, then generate stratified splits if `metadata.csv` is absent:
     ```bash
     python -m surveillance_tf.data.dcsass_loader \
       --data_root ./data/dcsass \
       --make_splits ./data/dcsass/splits
     ```
3. **Training (DCSASS default)**
   ```bash
   python -m surveillance_tf.train.train_mil_ucfcrime \
     --config ./configs/experiments/oneclass_dcsass.yaml \
     --out ./outputs/dcsass_run1
   ```
4. **Evaluation (DCSASS)**
   ```bash
   python -m surveillance_tf.train.eval_mil_ucfcrime \
     --data_root ./data/dcsass \
     --test_csv ./data/dcsass/splits/test.csv \
     --ckpt ./models/movinet/ckpt_best \
     --out ./outputs/dcsass_run1
   ```
5. **Demo**
   ```bash
   python -m surveillance_tf.demo.coord \
     --data_root ./data/dcsass \
     --video "./data/dcsass/sample/*.mp4" \
     --ckpt ./models/movinet/ckpt_best \
     --config ./configs/thresholds.yaml --fps 25
   ```
6. **Streamlit UI**
   ```bash
   python -m streamlit run surveillance_tf/demo/ui_app.py
   ```

## One-Class MIL (DCSASS)
The one-class MIL objective contrasts the mean of the top-k segment scores in each abnormal bag against the mean of the bottom-k scores, while sparsity and temporal smoothness regularisers encourage concise, coherent activations. This lets the model learn directly from anomalous videos without needing explicit normal footage. If you have curated normal/abnormal pairs you can still switch to the classic positive/negative MIL formulation by setting `--mil_mode posneg`.

Train One-Class MIL on DCSASS:
```bash
python -m surveillance_tf.train.train_mil_ucfcrime \
  --config ./configs/experiments/oneclass_dcsass.yaml \
  --out ./outputs/dcsass_ocmil
```

## Experiment Configuration
Hyperparameters can be centralised in YAML (see `configs/experiments/oneclass_dcsass.yaml`). Load them with `--config` and override any value from the command line:
```bash
python -m surveillance_tf.train.train_mil_ucfcrime \
  --config ./configs/experiments/oneclass_dcsass.yaml \
  --epochs 5 --lr 5e-5
```
The script automatically resolves dataset paths relative to the configuration file and writes TensorBoard logs plus the best checkpoint under `--out` (defaults to `./outputs/dcsass`).

## Data Validation
Before long training runs, verify that split CSVs reference valid videos:
```bash
python -m surveillance_tf.data.validate \
  --data_root ./data/dcsass \
  --splits train val test \
  --max_frames 1
```
The validator checks that each file exists, has non-zero size, and can be decoded via `imageio`. Use `--max_videos` for spot checks on large datasets or supply explicit CSV paths with `--csv`.

## Dataset Layout
```
data/
└── dcsass/
    ├── DCSASS Dataset/
    ├── sample/
    └── metadata.csv (optional)
```
## Repository Map
- `configs/` – Operating thresholds, camera metadata, class labels.
- `surveillance_tf/utils/` – Logging, seeding, metrics, geometric utilities, buffering, clip writers.
- `surveillance_tf/data/` – Video decoders, segment builders, tf.data loaders for UCF-Crime and DCSASS with CLI helpers.
- `surveillance_tf/models/` – MoViNet encoder and MIL scoring head.
- `surveillance_tf/losses/` – Ranking, sparsity, and smoothness losses.
- `surveillance_tf/train/` – Training and evaluation pipelines with SavedModel export hooks.
- `surveillance_tf/demo/` – SORT tracker, fusion, incident coordinator, SQLite DAO, responder utilities, Streamlit app, HTML templates.
- `surveillance_tf/export/` – SavedModel exporter CLI.
- `tests/` – Pytest-based unit and integration checks running on synthetic data.

## Citation
```
@inproceedings{sultani2018real,
  title     = {Real-World Anomaly Detection in Surveillance Videos},
  author    = {Sultani, Waqas and Chen, Chen and Shah, Mubarak},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2018}
}
```

## Safety & Privacy
- Only process footage you are authorised to handle; comply with local laws and organisational policy.
- Securely store datasets, incident logs, and generated reports; encrypt disks where possible.
- Review detection thresholds before deployment to minimise false alarms and privacy risks.
- Anonymise or redact personal data in exported clips or reports shared outside the security team.
