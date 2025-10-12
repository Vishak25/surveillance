"""MIL training script (DCSASS)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np

# Configure threading before TensorFlow import to avoid macOS mutex issues
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import tensorflow as tf

from surveillance_tf.data.dcsass_loader import load_split_entries, make_bag_dataset
from surveillance_tf.losses.mil_losses import compute_losses
from surveillance_tf.models.mil_head import MILScoringHead
from surveillance_tf.models.movinet_backbone import build_segment_encoder
from surveillance_tf.utils.logging import get_logger
from surveillance_tf.utils.metrics import roc_auc
from surveillance_tf.utils.paths import resolve_dcsass_root
from surveillance_tf.utils.seed import set_global_seed

LOGGER = get_logger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=None, help="Dataset root (defaults to ./data/dcsass).")
    parser.add_argument("--train_csv", type=Path, help="Optional CSV overriding the training split.")
    parser.add_argument("--val_csv", type=Path, help="Optional CSV overriding the validation split.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lambda_sparse", type=float, default=8e-5)
    parser.add_argument("--lambda_smooth", type=float, default=0.1)
    parser.add_argument("--freeze_backbone_until", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--stride", type=int, default=3)
    return parser.parse_args(list(argv) if argv is not None else None)


def _ensure_positive_negative(entries: List[dict]) -> None:
    labels = [int(entry["binary_label"]) for entry in entries]
    n_pos = sum(1 for label in labels if label == 1)
    n_neg = sum(1 for label in labels if label == 0)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training split must contain both positive and negative videos.")


def _default_split(root: Path, name: str) -> Path:
    return root / "splits" / f"{name}.csv"


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    set_global_seed(args.seed)

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    try:
        dataset_root = resolve_dcsass_root(args.data_root)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return

    train_csv = args.train_csv or _default_split(dataset_root, "train")
    val_csv = args.val_csv or _default_split(dataset_root, "val")

    train_entries = load_split_entries(dataset_root, "train", seed=args.seed, csv_path=train_csv)
    val_entries = load_split_entries(dataset_root, "val", seed=args.seed, csv_path=val_csv)
    _ensure_positive_negative(train_entries)

    train_ds = make_bag_dataset(
        dataset_root,
        split="train",
        csv_path=train_csv,
        T=args.T,
        stride=args.stride,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
        seed=args.seed,
    )
    val_ds = make_bag_dataset(
        dataset_root,
        split="val",
        csv_path=val_csv,
        T=args.T,
        stride=args.stride,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
        seed=args.seed,
    )

    train_ds = train_ds.map(lambda seg, label, vid: (seg, label, vid), num_parallel_calls=1)
    val_ds = val_ds.map(lambda seg, label, vid: (seg, label, vid), num_parallel_calls=1)

    encoder = build_segment_encoder(trainable=True)
    head = MILScoringHead()
    optimizer = tf.keras.optimizers.Adam(args.lr)

    n_pos = sum(1 for entry in train_entries if int(entry["binary_label"]) == 1)
    n_neg = sum(1 for entry in train_entries if int(entry["binary_label"]) == 0)

    pos_ds = train_ds.filter(lambda seg, label, vid: tf.equal(label, 1))
    neg_ds = train_ds.filter(lambda seg, label, vid: tf.equal(label, 0))
    pos_iter = iter(pos_ds.repeat())
    neg_iter = iter(neg_ds.repeat())

    out_dir = args.out
    log_dir = out_dir / "logs"
    model_dir = out_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    writer = tf.summary.create_file_writer(str(log_dir))
    best_auc = 0.0
    best_ckpt_path = model_dir / "ckpt_best"

    for epoch in range(1, args.epochs + 1):
        encoder.trainable = epoch > args.freeze_backbone_until
        steps = min(n_pos, n_neg)
        epoch_losses: List[float] = []
        for _ in range(steps):
            pos_segments, _, _ = next(pos_iter)
            neg_segments, _, _ = next(neg_iter)
            with tf.GradientTape() as tape:
                pos_features = encoder(pos_segments.to_tensor(), training=encoder.trainable)
                neg_features = encoder(neg_segments.to_tensor(), training=encoder.trainable)
                pos_scores = tf.squeeze(head(pos_features, training=True), axis=-1)
                neg_scores = tf.squeeze(head(neg_features, training=True), axis=-1)
                pos_rt = tf.RaggedTensor.from_tensor(tf.expand_dims(pos_scores, axis=0))
                neg_rt = tf.RaggedTensor.from_tensor(tf.expand_dims(neg_scores, axis=0))
                losses = compute_losses(
                    pos_rt,
                    neg_rt,
                    margin=args.margin,
                    lambda_sparse=args.lambda_sparse,
                    lambda_smooth=args.lambda_smooth,
                )
                loss_value = losses["total"]
            variables = encoder.trainable_variables + head.trainable_variables
            grads = tape.gradient(loss_value, variables)
            grads_and_vars = [(g, v) for g, v in zip(grads, variables) if g is not None]
            if grads_and_vars:
                optimizer.apply_gradients(grads_and_vars)
            epoch_losses.append(float(loss_value.numpy()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        LOGGER.info("Epoch %d/%d: mean loss %.5f", epoch, args.epochs, mean_loss)

        val_labels: List[int] = []
        val_scores: List[float] = []
        for segments, label, _ in val_ds:
            features = encoder(segments.to_tensor(), training=False)
            scores = tf.squeeze(head(features, training=False), axis=-1)
            score = float(tf.reduce_max(scores).numpy()) if tf.size(scores) > 0 else 0.0
            val_labels.append(int(label.numpy()))
            val_scores.append(score)
        val_auc = roc_auc(np.array(val_labels), np.array(val_scores))
        with writer.as_default():
            tf.summary.scalar("val_auc", val_auc, step=epoch)
            tf.summary.scalar("train_loss", mean_loss, step=epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            LOGGER.info("New best AUC %.4f at epoch %d", best_auc, epoch)
            seg_input = tf.keras.Input(shape=(None, args.image_size[0], args.image_size[1], 3), name="segments")
            feats = encoder(seg_input)
            logits = head(feats)
            infer_model = tf.keras.Model(inputs=seg_input, outputs=logits, name="mil_inference")
            tf.saved_model.save(infer_model, str(best_ckpt_path))
            shared_dir = Path("models/movinet/ckpt_best")
            shared_dir.parent.mkdir(parents=True, exist_ok=True)
            tf.saved_model.save(infer_model, str(shared_dir))
            state = {"epoch": epoch, "val_auc": best_auc, "encoder_trainable": encoder.trainable}
            with (model_dir / "training_state.json").open("w", encoding="utf-8") as fp:
                json.dump(state, fp, indent=2)

    LOGGER.info("Training complete. Best AUC %.4f. Saved to %s", best_auc, best_ckpt_path)


if __name__ == "__main__":
    main()
