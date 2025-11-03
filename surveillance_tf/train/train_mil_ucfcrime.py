"""Simplified MIL training loop tailored for the DCSASS dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import yaml

# Configure threading before TensorFlow import to avoid macOS mutex issues
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

# Set ffmpeg path for imageio (Homebrew on Apple Silicon installs to /opt/homebrew)
if os.path.exists("/opt/homebrew/bin/ffmpeg"):
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "/opt/homebrew/bin/ffmpeg")
elif os.path.exists("/usr/local/bin/ffmpeg"):
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "/usr/local/bin/ffmpeg")

import tensorflow as tf

from surveillance_tf.data.dcsass_loader import (
    load_split_entries,
    make_bag_dataset,
    make_clip_dataset,
)
from surveillance_tf.losses.mil_losses import compute_losses, compute_one_class_mil_losses
from surveillance_tf.models.mil_head import MILScoringHead
from surveillance_tf.models.movinet_backbone import build_segment_encoder
from surveillance_tf.utils.logging import get_logger
from surveillance_tf.utils.metrics import roc_auc
from surveillance_tf.utils.paths import resolve_dcsass_root
from surveillance_tf.utils.seed import set_global_seed
from tqdm.auto import tqdm

LOGGER = get_logger(__name__)


DEFAULT_CONFIG_PATH = Path("configs/experiments/oneclass_dcsass.yaml")


@dataclass
class TrainingConfig:
    """Configuration describing a training run loaded from YAML."""

    data_root: Optional[Path] = None
    train_csv: Optional[Path] = None
    val_csv: Optional[Path] = None
    out: Optional[Path] = None
    epochs: Optional[int] = None
    lr: Optional[float] = None
    trainer: Optional[str] = None
    mil_mode: Optional[str] = None
    k: Optional[int] = None
    margin: Optional[float] = None
    lambda_sparse: Optional[float] = None
    lambda_smooth: Optional[float] = None
    freeze_backbone_until: Optional[int] = None
    seed: Optional[int] = None
    batch_size: Optional[int] = None
    image_size: Optional[Sequence[int]] = None
    T: Optional[int] = None
    stride: Optional[int] = None
    class_loss_weight: Optional[float] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML config must contain a mapping. Got {type(data)!r} from {path}.")
        config = cls()
        config.update_from_dict(data)
        return config

    def update_from_dict(self, overrides: dict) -> None:
        if not overrides:
            return
        path_fields = {"data_root", "train_csv", "val_csv", "out"}
        int_fields = {"epochs", "k", "freeze_backbone_until", "seed", "batch_size", "T", "stride"}
        float_fields = {"lr", "margin", "lambda_sparse", "lambda_smooth", "class_loss_weight"}
        for field_info in fields(self):
            name = field_info.name
            if name not in overrides:
                continue
            value = overrides[name]
            if value is None:
                continue
            if name in path_fields:
                setattr(self, name, Path(value))
            elif name == "image_size":
                setattr(self, name, tuple(int(v) for v in value))
            elif name in int_fields:
                setattr(self, name, int(value))
            elif name in float_fields:
                setattr(self, name, float(value))
            else:
                setattr(self, name, value)

    def apply_cli(self, args: argparse.Namespace) -> None:
        cli_overrides = {
            key: getattr(args, key)
            for key in {
                "data_root",
                "train_csv",
                "val_csv",
                "out",
                "epochs",
                "lr",
                "trainer",
                "mil_mode",
                "k",
                "margin",
                "lambda_sparse",
                "lambda_smooth",
                "freeze_backbone_until",
                "seed",
                "batch_size",
                "image_size",
                "T",
                "stride",
                "class_loss_weight",
            }
            if getattr(args, key) is not None
        }
        self.update_from_dict(cli_overrides)

    def ensure_complete(self) -> None:
        if self.trainer is None:
            self.trainer = "mil"
        else:
            self.trainer = str(self.trainer).lower()

        if self.seed is None:
            self.seed = 1337
        else:
            self.seed = int(self.seed)

        common_required = [
            "epochs",
            "lr",
            "batch_size",
            "image_size",
            "T",
            "stride",
            "out",
        ]
        missing = [name for name in common_required if getattr(self, name) is None]
        if missing:
            raise ValueError(f"Training configuration missing required fields: {', '.join(missing)}")

        if self.trainer == "mil":
            mil_required = ["mil_mode", "k", "margin", "lambda_sparse", "lambda_smooth", "freeze_backbone_until"]
            mil_missing = [name for name in mil_required if getattr(self, name) is None]
            if mil_missing:
                raise ValueError(
                    f"MIL trainer selected but configuration missing fields: {', '.join(mil_missing)}"
                )
            self.mil_mode = str(self.mil_mode).lower()
        elif self.trainer == "supervised":
            if self.class_loss_weight is None:
                self.class_loss_weight = 0.0
        else:
            raise ValueError(f"Unsupported trainer '{self.trainer}'.")

        if self.class_loss_weight is not None:
            self.class_loss_weight = float(self.class_loss_weight)

    def resolve_paths(self) -> None:
        root = resolve_dcsass_root(self.data_root)
        self.data_root = root
        if self.train_csv is None:
            self.train_csv = root / "splits" / "train.csv"
        else:
            train_path = Path(self.train_csv)
            self.train_csv = (train_path if train_path.is_absolute() else (root / train_path)).resolve()
        if self.val_csv is None:
            self.val_csv = root / "splits" / "val.csv"
        else:
            val_path = Path(self.val_csv)
            self.val_csv = (val_path if val_path.is_absolute() else (root / val_path)).resolve()
        if self.out is None:
            raise ValueError("Output directory must be specified via configuration or CLI.")
        self.out = Path(self.out).expanduser().resolve()

    @property
    def image_size_tuple(self) -> tuple[int, int]:
        if self.image_size is None:
            raise ValueError("image_size must be provided in the configuration.")
        if len(self.image_size) != 2:
            raise ValueError("image_size must contain exactly two values (height, width).")
        return int(self.image_size[0]), int(self.image_size[1])


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML experiment description.")
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--train_csv", type=Path, default=None)
    parser.add_argument("--val_csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--trainer", choices=("mil", "supervised"), default=None)
    parser.add_argument("--mil_mode", choices=("oneclass", "posneg"), default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--lambda_sparse", type=float, default=None)
    parser.add_argument("--lambda_smooth", type=float, default=None)
    parser.add_argument("--freeze_backbone_until", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--class_loss_weight", type=float, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def _build_dataset_iterator(dataset: tf.data.Dataset) -> Iterator:
    return iter(dataset.repeat())


def _maybe_squeeze_first_dim(tensor: tf.Tensor) -> tf.Tensor:
    rank_static = tensor.shape.rank
    if rank_static is not None:
        if rank_static == 0:
            return tensor
        if tensor.shape[0] == 1:
            return tf.squeeze(tensor, axis=0)
        return tensor
    return tf.cond(
        tf.logical_and(tf.greater_equal(tf.rank(tensor), 1), tf.equal(tf.shape(tensor)[0], 1)),
        lambda: tf.squeeze(tensor, axis=0),
        lambda: tensor,
    )


def _ensure_batch_dim(tensor: tf.Tensor) -> tf.Tensor:
    rank_static = tensor.shape.rank
    if rank_static is not None:
        if rank_static == 1:
            return tensor[tf.newaxis, :]
        return tensor
    return tf.cond(
        tf.equal(tf.rank(tensor), 1),
        lambda: tf.expand_dims(tensor, 0),
        lambda: tensor,
    )


class ClipClassificationHead(tf.keras.Model):
    """Dense head producing anomaly and optional class predictions."""

    def __init__(
        self,
        num_classes: int,
        hidden_units: Sequence[int] | None = None,
        dropout_rate: float = 0.3,
        name: str = "clip_head",
    ) -> None:
        super().__init__(name=name)
        hidden_units = list(hidden_units or (256, 128))
        self.hidden_layers: list[tf.keras.layers.Layer] = []
        for idx, units in enumerate(hidden_units):
            self.hidden_layers.append(tf.keras.layers.Dense(units, activation="relu", name=f"dense_{idx}"))
            self.hidden_layers.append(tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{idx}"))
        self.anomaly_head = tf.keras.layers.Dense(1, activation="sigmoid", name="anomaly")
        self.class_head = (
            tf.keras.layers.Dense(num_classes, activation="softmax", name="class") if num_classes > 1 else None
        )

    def call(self, inputs: tf.Tensor, training: bool = False):  # type: ignore[override]
        x = inputs
        for layer in self.hidden_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        outputs = {"anomaly": self.anomaly_head(x)}
        if self.class_head is not None:
            outputs["class"] = self.class_head(x)
        return outputs


def _collect_scores(model_encoder, model_head, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    labels: List[int] = []
    scores: List[float] = []
    for segments, label, _ in dataset:
        seg_tensor = segments.to_tensor() if isinstance(segments, tf.RaggedTensor) else tf.convert_to_tensor(segments)
        seg_tensor = _maybe_squeeze_first_dim(seg_tensor)
        features = model_encoder(seg_tensor, training=False)
        raw_scores = tf.squeeze(model_head(features, training=False), axis=-1)
        score = float(tf.reduce_max(raw_scores).numpy()) if tf.size(raw_scores) > 0 else 0.0
        labels.append(int(label.numpy()))
        scores.append(score)
    return np.array(labels), np.array(scores)


def _ensure_positive_negative(entries: Sequence[dict]) -> None:
    labels = [int(entry["binary_label"]) for entry in entries]
    if not any(label == 1 for label in labels) or not any(label == 0 for label in labels):
        raise ValueError("Training split must contain both positive and negative videos.")


def _prepare_mil_datasets(config: TrainingConfig):
    entries = load_split_entries(config.data_root, "train", seed=config.seed, csv_path=config.train_csv)

    if config.mil_mode == "oneclass":
        abnormal = [entry for entry in entries if int(entry.get("binary_label", 1)) == 1]
        if not abnormal:
            raise RuntimeError("One-Class MIL requires at least one abnormal training video.")
        LOGGER.info("Loaded %d abnormal training videos for One-Class MIL.", len(abnormal))
        train_ds = make_bag_dataset(
            config.data_root,
            split="train",
            entries=abnormal,
            T=config.T,
            stride=config.stride,
            batch_size=1,
            image_size=config.image_size_tuple,
            seed=config.seed,
        )
        train_ds = train_ds.map(lambda segments, *_: segments, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        steps = len(abnormal)
        pos_iter: Optional[Iterator] = _build_dataset_iterator(train_ds)
        neg_iter: Optional[Iterator] = None
    else:
        _ensure_positive_negative(entries)
        base_ds = make_bag_dataset(
            config.data_root,
            split="train",
            csv_path=config.train_csv,
            T=config.T,
            stride=config.stride,
            batch_size=config.batch_size,
            image_size=config.image_size_tuple,
            seed=config.seed,
        ).map(lambda seg, label, vid: (seg, label, vid), num_parallel_calls=tf.data.AUTOTUNE)
        base_ds = base_ds.prefetch(tf.data.AUTOTUNE)
        pos_ds = base_ds.filter(lambda seg, label, vid: tf.equal(label, 1))
        neg_ds = base_ds.filter(lambda seg, label, vid: tf.equal(label, 0))
        pos_count = sum(1 for entry in entries if int(entry["binary_label"]) == 1)
        neg_count = sum(1 for entry in entries if int(entry["binary_label"]) == 0)
        steps = min(pos_count, neg_count)
        pos_iter = _build_dataset_iterator(pos_ds)
        neg_iter = _build_dataset_iterator(neg_ds)

    val_ds = make_bag_dataset(
        config.data_root,
        split="val",
        csv_path=config.val_csv,
        T=config.T,
        stride=config.stride,
        batch_size=config.batch_size,
        image_size=config.image_size_tuple,
        seed=config.seed,
    ).map(lambda seg, label, vid: (seg, label, vid), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    if steps == 0:
        raise RuntimeError("Training split does not contain enough samples to form a batch.")

    return steps, pos_iter, neg_iter, val_ds


def _prepare_supervised_datasets(config: TrainingConfig):
    train_entries = load_split_entries(config.data_root, "train", seed=config.seed, csv_path=config.train_csv)
    val_entries = load_split_entries(config.data_root, "val", seed=config.seed, csv_path=config.val_csv)

    if not train_entries:
        raise RuntimeError("Training split is empty; cannot start supervised training.")

    train_ds, train_entries = make_clip_dataset(
        config.data_root,
        split="train",
        entries=train_entries,
        T=config.T,
        batch_size=config.batch_size,
        image_size=config.image_size_tuple,
        seed=config.seed,
        shuffle=True,
        repeat=True,
    )
    val_ds, val_entries = make_clip_dataset(
        config.data_root,
        split="val",
        entries=val_entries,
        T=config.T,
        batch_size=config.batch_size,
        image_size=config.image_size_tuple,
        seed=config.seed,
        shuffle=False,
        repeat=False,
    )

    steps_per_epoch = math.ceil(len(train_entries) / max(config.batch_size, 1))
    if steps_per_epoch == 0:
        raise RuntimeError("Batch size exceeds number of training clips; adjust batch_size or dataset.")

    num_classes = max(entry.get("class_index", entry.get("label_index", 0)) for entry in train_entries) + 1
    return train_ds, val_ds, steps_per_epoch, num_classes


def _train_oneclass_step(encoder, head, optimizer, bag, train_backbone: bool, config: TrainingConfig):
    segments = bag
    segments = segments.to_tensor() if isinstance(segments, tf.RaggedTensor) else tf.convert_to_tensor(segments)
    segments = _maybe_squeeze_first_dim(segments)
    with tf.GradientTape() as tape:
        features = encoder(segments, training=train_backbone)
        raw_scores = tf.squeeze(head(features, training=True), axis=-1)
        raw_scores = _ensure_batch_dim(raw_scores)
        ragged_scores = tf.RaggedTensor.from_tensor(raw_scores)
        losses = compute_one_class_mil_losses(
            bag_scores=ragged_scores,
            k=config.k,
            margin=config.margin,
            lambda_sparse=config.lambda_sparse,
            lambda_smooth=config.lambda_smooth,
        )
        total = losses["total"]
    variables = encoder.trainable_variables + head.trainable_variables
    grads = tape.gradient(total, variables)
    optimizer.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])
    return {name: float(value.numpy()) for name, value in losses.items()}


def _train_posneg_step(encoder, head, optimizer, pos_batch, neg_batch, train_backbone: bool, config: TrainingConfig):
    pos_segments = pos_batch[0]
    neg_segments = neg_batch[0]
    pos_segments = pos_segments.to_tensor() if isinstance(pos_segments, tf.RaggedTensor) else tf.convert_to_tensor(pos_segments)
    neg_segments = neg_segments.to_tensor() if isinstance(neg_segments, tf.RaggedTensor) else tf.convert_to_tensor(neg_segments)
    pos_segments = _maybe_squeeze_first_dim(pos_segments)
    neg_segments = _maybe_squeeze_first_dim(neg_segments)
    with tf.GradientTape() as tape:
        pos_features = encoder(pos_segments, training=train_backbone)
        neg_features = encoder(neg_segments, training=train_backbone)
        pos_scores = tf.squeeze(head(pos_features, training=True), axis=-1)
        neg_scores = tf.squeeze(head(neg_features, training=True), axis=-1)
        pos_scores = _ensure_batch_dim(pos_scores)
        neg_scores = _ensure_batch_dim(neg_scores)
        pos_rt = tf.RaggedTensor.from_tensor(pos_scores)
        neg_rt = tf.RaggedTensor.from_tensor(neg_scores)
        losses = compute_losses(
            pos_rt,
            neg_rt,
            margin=config.margin,
            lambda_sparse=config.lambda_sparse,
            lambda_smooth=config.lambda_smooth,
        )
        total = losses["total"]
    variables = encoder.trainable_variables + head.trainable_variables
    grads = tape.gradient(total, variables)
    optimizer.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])
    return {name: float(value.numpy()) for name, value in losses.items()}


def _train_supervised_step(
    encoder,
    head,
    optimizer,
    batch,
    train_backbone: bool,
    binary_loss_fn,
    class_loss_fn,
    class_loss_weight: float,
):
    clips, binary_labels, class_labels, _ = batch
    binary_labels = tf.cast(binary_labels, tf.float32)
    class_labels = tf.cast(class_labels, tf.int32)
    with tf.GradientTape() as tape:
        features = encoder(clips, training=train_backbone)
        outputs = head(features, training=True)
        anomaly_logits = tf.squeeze(outputs["anomaly"], axis=-1)
        anomaly_loss = binary_loss_fn(binary_labels, anomaly_logits)
        total_loss = anomaly_loss
        class_loss_value = None
        if "class" in outputs and class_loss_fn is not None:
            class_loss_value = class_loss_fn(class_labels, outputs["class"])
            total_loss = total_loss + class_loss_weight * class_loss_value
    variables = list(head.trainable_variables)
    if train_backbone:
        variables += encoder.trainable_variables
    grads = tape.gradient(total_loss, variables)
    optimizer.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])
    return {
        "total": total_loss,
        "anomaly": anomaly_loss,
        "class": class_loss_value,
        "outputs": outputs,
        "binary_labels": binary_labels,
        "class_labels": class_labels,
    }


def _save_checkpoint(encoder, head, image_size: tuple[int, int], checkpoint_dir: Path, epoch: int, val_auc: float) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seg_input = tf.keras.Input(shape=(None, image_size[0], image_size[1], 3), name="segments")
    features = encoder(seg_input)
    logits = head(features)
    model = tf.keras.Model(inputs=seg_input, outputs=logits, name="mil_inference")
    tf.saved_model.save(model, str(checkpoint_dir / "ckpt_best"))
    shared_dir = Path("models/movinet/ckpt_best")
    shared_dir.parent.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(shared_dir))
    state = {"epoch": epoch, "val_auc": val_auc, "encoder_trainable": encoder.trainable}
    with (checkpoint_dir / "training_state.json").open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _save_supervised_checkpoint(
    encoder,
    head,
    image_size: tuple[int, int],
    T: int,
    checkpoint_dir: Path,
    epoch: int,
    val_auc: float,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    clip_input = tf.keras.Input(shape=(T, image_size[0], image_size[1], 3), name="clip")
    features = encoder(clip_input)
    outputs = head(features)
    model = tf.keras.Model(inputs=clip_input, outputs=outputs, name="clip_classifier")
    tf.saved_model.save(model, str(checkpoint_dir / "ckpt_best"))
    shared_dir = Path("models/movinet/ckpt_best")
    shared_dir.parent.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(shared_dir))
    state = {"epoch": epoch, "val_auc": val_auc, "encoder_trainable": encoder.trainable}
    with (checkpoint_dir / "training_state.json").open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _run_supervised_training(config: TrainingConfig, log_dir: Path, checkpoints_dir: Path) -> None:
    train_ds, val_ds, steps_per_epoch, num_classes = _prepare_supervised_datasets(config)
    encoder = build_segment_encoder(trainable=True)
    head = ClipClassificationHead(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(config.lr)
    binary_loss_fn = tf.keras.losses.BinaryCrossentropy()
    class_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy() if num_classes > 1 else None

    summary_writer = tf.summary.create_file_writer(str(log_dir))
    best_auc = 0.0
    global_step = 0
    train_iter = iter(train_ds)

    for epoch in range(1, config.epochs + 1):
        encoder.trainable = epoch > config.freeze_backbone_until
        LOGGER.info(
            "Starting epoch %d/%d for supervised training (backbone trainable=%s)",
            epoch,
            config.epochs,
            encoder.trainable,
        )

        epoch_totals: List[float] = []
        epoch_anomaly: List[float] = []
        epoch_class: List[float] = []
        train_auc_metric = tf.keras.metrics.AUC()
        train_bin_acc = tf.keras.metrics.BinaryAccuracy()
        train_cls_acc = tf.keras.metrics.SparseCategoricalAccuracy() if num_classes > 1 else None

        progress = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch}/{config.epochs}",
            unit="step",
            leave=False,
        )
        for _ in progress:
            batch = next(train_iter)
            result = _train_supervised_step(
                encoder,
                head,
                optimizer,
                batch,
                encoder.trainable,
                binary_loss_fn,
                class_loss_fn,
                config.class_loss_weight,
            )
            outputs = result["outputs"]
            bin_labels = result["binary_labels"]
            class_labels = result["class_labels"]
            bin_probs = tf.squeeze(outputs["anomaly"], axis=-1)
            train_auc_metric.update_state(bin_labels, bin_probs)
            train_bin_acc.update_state(bin_labels, bin_probs)
            total_loss = float(result["total"].numpy())
            anomaly_loss = float(result["anomaly"].numpy())
            epoch_totals.append(total_loss)
            epoch_anomaly.append(anomaly_loss)
            if result["class"] is not None:
                class_loss_value = float(result["class"].numpy())
                epoch_class.append(class_loss_value)
                if train_cls_acc is not None:
                    train_cls_acc.update_state(class_labels, outputs["class"])
            global_step += 1
            progress.set_postfix(total=f"{total_loss:.4f}")
            with summary_writer.as_default():
                tf.summary.scalar("train/supervised_step_total", total_loss, step=global_step)
                tf.summary.scalar("train/supervised_step_anomaly", anomaly_loss, step=global_step)
                if result["class"] is not None:
                    tf.summary.scalar("train/supervised_step_class", class_loss_value, step=global_step)
        progress.close()

        mean_total = float(np.mean(epoch_totals)) if epoch_totals else float("nan")
        LOGGER.info("Epoch %d supervised mean loss %.5f", epoch, mean_total)
        with summary_writer.as_default():
            tf.summary.scalar("train/supervised_epoch_total", mean_total, step=epoch)
            tf.summary.scalar(
                "train/supervised_epoch_anomaly",
                float(np.mean(epoch_anomaly)) if epoch_anomaly else float("nan"),
                step=epoch,
            )
            if epoch_class:
                tf.summary.scalar("train/supervised_epoch_class", float(np.mean(epoch_class)), step=epoch)
            tf.summary.scalar("train/supervised_auc", train_auc_metric.result(), step=epoch)
            tf.summary.scalar("train/supervised_bin_acc", train_bin_acc.result(), step=epoch)
            if train_cls_acc is not None:
                tf.summary.scalar("train/supervised_cls_acc", train_cls_acc.result(), step=epoch)

        val_anomaly_losses: List[float] = []
        val_class_losses: List[float] = []
        val_totals: List[float] = []
        val_bin_acc = tf.keras.metrics.BinaryAccuracy()
        val_cls_acc = tf.keras.metrics.SparseCategoricalAccuracy() if num_classes > 1 else None
        val_labels: List[int] = []
        val_scores: List[float] = []
        for clips, bin_labels, class_labels, _ in val_ds:
            bin_labels = tf.cast(bin_labels, tf.float32)
            features = encoder(clips, training=False)
            outputs = head(features, training=False)
            bin_probs = tf.squeeze(outputs["anomaly"], axis=-1)
            anomaly_loss = float(binary_loss_fn(bin_labels, bin_probs).numpy())
            total_loss = anomaly_loss
            val_anomaly_losses.append(anomaly_loss)
            val_bin_acc.update_state(bin_labels, bin_probs)
            val_labels.extend(bin_labels.numpy().astype(int).tolist())
            val_scores.extend(bin_probs.numpy().tolist())
            if "class" in outputs and class_loss_fn is not None:
                class_loss = float(class_loss_fn(class_labels, outputs["class"]).numpy())
                val_class_losses.append(class_loss)
                if val_cls_acc is not None:
                    val_cls_acc.update_state(class_labels, outputs["class"])
                total_loss += config.class_loss_weight * class_loss
            val_totals.append(total_loss)

        val_auc = None
        if val_scores and len(set(val_labels)) >= 2:
            val_auc = roc_auc(np.array(val_labels), np.array(val_scores))
            with summary_writer.as_default():
                tf.summary.scalar("val/supervised_auc", val_auc, step=epoch)
            LOGGER.info("Epoch %d validation AUC: %.4f", epoch, val_auc)
        else:
            LOGGER.warning("Validation split lacks class diversity; skipping AUC computation.")

        with summary_writer.as_default():
            tf.summary.scalar(
                "val/supervised_total",
                float(np.mean(val_totals)) if val_totals else float("nan"),
                step=epoch,
            )
            tf.summary.scalar(
                "val/supervised_anomaly",
                float(np.mean(val_anomaly_losses)) if val_anomaly_losses else float("nan"),
                step=epoch,
            )
            if val_class_losses:
                tf.summary.scalar(
                    "val/supervised_class",
                    float(np.mean(val_class_losses)),
                    step=epoch,
                )
            tf.summary.scalar("val/supervised_bin_acc", val_bin_acc.result(), step=epoch)
            if val_cls_acc is not None:
                tf.summary.scalar("val/supervised_cls_acc", val_cls_acc.result(), step=epoch)

        if val_auc is not None and val_auc >= best_auc:
            best_auc = val_auc
            LOGGER.info("New best supervised validation AUC %.4f at epoch %d", best_auc, epoch)
            _save_supervised_checkpoint(encoder, head, config.image_size_tuple, config.T, checkpoints_dir, epoch, best_auc)

    LOGGER.info("Supervised training finished. Best validation AUC %s", f"{best_auc:.4f}" if best_auc else "N/A")


def run_training(config: TrainingConfig) -> None:
    config.ensure_complete()
    set_global_seed(config.seed)
    config.resolve_paths()

    if config.trainer not in {"mil", "supervised"}:
        raise ValueError(f"Unsupported trainer '{config.trainer}'.")
    if config.trainer == "mil" and config.mil_mode not in {"oneclass", "posneg"}:
        raise ValueError(f"Unsupported MIL mode '{config.mil_mode}'.")

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    config.out.mkdir(parents=True, exist_ok=True)
    log_dir = config.out / "logs"
    checkpoints_dir = config.out / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Selected trainer '%s'", config.trainer)
    if config.trainer == "mil":
        LOGGER.info("MIL mode '%s'", config.mil_mode)

    if config.trainer == "supervised":
        _run_supervised_training(config, log_dir, checkpoints_dir)
        return

    steps_per_epoch, pos_iter, neg_iter, val_ds = _prepare_mil_datasets(config)

    encoder = build_segment_encoder(trainable=True)
    head = MILScoringHead()
    optimizer = tf.keras.optimizers.Adam(config.lr)

    summary_writer = tf.summary.create_file_writer(str(log_dir))
    best_auc = 0.0
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        encoder.trainable = epoch > config.freeze_backbone_until
        LOGGER.info("Starting epoch %d/%d (backbone trainable=%s)", epoch, config.epochs, encoder.trainable)

        epoch_totals: List[float] = []
        epoch_rank: List[float] = []
        epoch_sparse: List[float] = []
        epoch_smooth: List[float] = []

        progress = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch}/{config.epochs}",
            unit="step",
            leave=False,
        )
        if config.mil_mode == "oneclass":
            assert pos_iter is not None
            for _ in progress:
                losses = _train_oneclass_step(encoder, head, optimizer, next(pos_iter), encoder.trainable, config)
                epoch_totals.append(losses["total"])
                epoch_rank.append(losses["ranking"])
                epoch_sparse.append(losses["sparsity"])
                epoch_smooth.append(losses["smoothness"])
                global_step += 1
                progress.set_postfix(total=f"{losses['total']:.4f}")
                with summary_writer.as_default():
                    tf.summary.scalar("train/ocmil_total", losses["total"], step=global_step)
                    tf.summary.scalar("train/ocmil_ranking", losses["ranking"], step=global_step)
                    tf.summary.scalar("train/ocmil_sparsity", losses["sparsity"], step=global_step)
                    tf.summary.scalar("train/ocmil_smoothness", losses["smoothness"], step=global_step)
        else:
            assert pos_iter is not None and neg_iter is not None
            for _ in progress:
                losses = _train_posneg_step(encoder, head, optimizer, next(pos_iter), next(neg_iter), encoder.trainable, config)
                epoch_totals.append(losses["total"])
                global_step += 1
                progress.set_postfix(total=f"{losses['total']:.4f}")
                with summary_writer.as_default():
                    tf.summary.scalar("train/total_loss", losses["total"], step=global_step)
        progress.close()

        mean_total = float(np.mean(epoch_totals)) if epoch_totals else float("nan")
        LOGGER.info("Epoch %d mean loss %.5f", epoch, mean_total)
        with summary_writer.as_default():
            tf.summary.scalar("train/epoch_total_loss", mean_total, step=epoch)
            if epoch_rank:
                tf.summary.scalar("train/epoch_ranking", float(np.mean(epoch_rank)), step=epoch)
            if epoch_sparse:
                tf.summary.scalar("train/epoch_sparsity", float(np.mean(epoch_sparse)), step=epoch)
            if epoch_smooth:
                tf.summary.scalar("train/epoch_smoothness", float(np.mean(epoch_smooth)), step=epoch)

        val_labels, val_scores = _collect_scores(encoder, head, val_ds)
        with summary_writer.as_default():
            tf.summary.scalar("val/score_mean", float(np.mean(val_scores)) if val_scores.size else float("nan"), step=epoch)
            tf.summary.scalar("val/score_std", float(np.std(val_scores)) if val_scores.size else float("nan"), step=epoch)

        val_auc: Optional[float] = None
        if np.unique(val_labels).size >= 2 and val_scores.size:
            val_auc = roc_auc(val_labels, val_scores)
            with summary_writer.as_default():
                tf.summary.scalar("val/auc", val_auc, step=epoch)
            LOGGER.info("Epoch %d validation AUC: %.4f", epoch, val_auc)
        else:
            LOGGER.warning("Validation split contains a single class; skipping ROC-AUC computation.")

        if val_auc is not None and val_auc >= best_auc:
            best_auc = val_auc
            LOGGER.info("New best validation AUC %.4f at epoch %d", best_auc, epoch)
            _save_checkpoint(encoder, head, config.image_size_tuple, checkpoints_dir, epoch, best_auc)

    LOGGER.info("Training finished. Best validation AUC %.4f", best_auc)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.config:
        config_path = args.config.expanduser().resolve()
    else:
        config_path = DEFAULT_CONFIG_PATH
        LOGGER.info("No --config supplied; defaulting to %s", config_path)
    config = TrainingConfig.from_yaml(config_path)
    config.apply_cli(args)
    config.ensure_complete()
    run_training(config)


if __name__ == "__main__":
    main()
