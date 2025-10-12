"""Evaluation script for DCSASS MIL model."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

import numpy as np
import tensorflow as tf

from surveillance_tf.data.dcsass_loader import make_bag_dataset
from surveillance_tf.utils.logging import get_logger
from surveillance_tf.utils.metrics import histogram_scores, plot_roc, roc_auc
from surveillance_tf.utils.paths import resolve_dcsass_root
from surveillance_tf.utils.seed import set_global_seed

LOGGER = get_logger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=None, help="Dataset root (defaults to ./data/dcsass).")
    parser.add_argument("--test_csv", type=Path, help="Optional CSV overriding the test split.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mil_mode", choices=("oneclass", "posneg"), default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--image_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--stride", type=int, default=3)
    return parser.parse_args(list(argv) if argv is not None else None)


def _default_split(root: Path) -> Path:
    return root / "splits" / "test.csv"


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    set_global_seed(args.seed)

    if not args.ckpt.exists():
        LOGGER.error("Checkpoint not found: %s", args.ckpt)
        return

    try:
        dataset_root = resolve_dcsass_root(args.data_root)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return

    test_csv = args.test_csv or _default_split(dataset_root)
    if not Path(test_csv).exists():
        LOGGER.error("Test CSV not found: %s", test_csv)
        return

    ds = make_bag_dataset(
        dataset_root,
        split="test",
        csv_path=test_csv,
        T=args.T,
        stride=args.stride,
        batch_size=1,
        image_size=tuple(args.image_size),
        seed=args.seed,
    ).map(lambda seg, label, vid: (seg, label, vid))

    labels: List[int] = []
    scores: List[float] = []

    model = tf.keras.models.load_model(str(args.ckpt))

    for segments, label, _ in ds:
        outputs = model(segments.to_tensor(), training=False)
        outputs = tf.squeeze(outputs, axis=-1)
        score = float(tf.reduce_max(outputs).numpy()) if tf.size(outputs) > 0 else 0.0
        labels.append(int(label.numpy()))
        scores.append(score)

    if not scores:
        LOGGER.error("No samples evaluated. Verify dataset and split configuration.")
        return

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_arr = np.array(labels)
    scores_arr = np.array(scores)
    unique_labels = np.unique(labels_arr)
    auc_value: float | None = None
    if unique_labels.size >= 2:
        auc_value = roc_auc(labels_arr, scores_arr)
        plot_roc(labels_arr, scores_arr, out_dir / "roc_curve.png")
    else:
        LOGGER.warning(
            "Evaluation labels contain a single class (%s); skipping ROC-AUC.",
            unique_labels[0] if unique_labels.size == 1 else "unknown",
        )
    histogram_scores(scores_arr, out_dir / "score_hist.png")

    with (out_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "score"])
        writer.writerows(zip(labels, scores))

    if auc_value is not None:
        LOGGER.info("Evaluation complete. AUC=%.4f", auc_value)
    else:
        LOGGER.info("Evaluation complete. Scores written to %s (AUC skipped).", out_dir / "scores.csv")


if __name__ == "__main__":
    main()
