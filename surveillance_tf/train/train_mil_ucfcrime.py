"""Simplified MIL training loop tailored for the DCSASS dataset."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, get_args, get_origin

import numpy as np
import yaml

# Configure threading before TensorFlow import to avoid macOS mutex issues
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import tensorflow as tf

from surveillance_tf.data.dcsass_loader import load_split_entries, make_bag_dataset
from surveillance_tf.losses.mil_losses import compute_losses, compute_one_class_mil_losses
from surveillance_tf.models.mil_head import MILScoringHead
from surveillance_tf.models.movinet_backbone import build_segment_encoder
from surveillance_tf.utils.logging import get_logger
from surveillance_tf.utils.metrics import roc_auc
from surveillance_tf.utils.paths import resolve_dcsass_root
from surveillance_tf.utils.seed import set_global_seed

LOGGER = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration describing a training run.

    The defaults match the hyper-parameters reported in the reference paper and
    can be overridden via YAML or CLI flags.  Paths are resolved lazily so that
    they can be expressed relative to the project root inside configuration
    files.
    """

    data_root: Path = field(default_factory=lambda: Path("./data/dcsass"))
    train_csv: Optional[Path] = None
    val_csv: Optional[Path] = None
    out: Path = field(default_factory=lambda: Path("./outputs/dcsass"))
    epochs: int = 10
    lr: float = 1e-4
    mil_mode: str = "oneclass"
    k: int = 3
    margin: float = 1.0
    lambda_sparse: float = 8e-5
    lambda_smooth: float = 0.1
    freeze_backbone_until: int = 1
    seed: int = 1337
    batch_size: int = 1
    image_size: Sequence[int] = field(default_factory=lambda: (224, 224))
    T: int = 32
    stride: int = 3

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML config must contain a mapping. Got {type(data)!r} from {path}.")
        config = cls()
        config.update_from_dict(data)
        return config

    def update_from_dict(self, overrides: dict) -> None:
        for field_info in fields(self):
            name = field_info.name
            if name not in overrides:
                continue
            value = overrides[name]
            if value is None:
                continue
            field_type = field_info.type
            origin = get_origin(field_type)
            args = get_args(field_type)
            is_path = field_type is Path or (origin is not None and Path in args)
            if is_path:
                setattr(self, name, Path(value))
            elif name == "image_size":
                setattr(self, name, tuple(int(v) for v in value))
            elif field_info.type is int:
                setattr(self, name, int(value))
            elif field_info.type is float:
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
            }
            if getattr(args, key) is not None
        }
        self.update_from_dict(cli_overrides)

    def resolve_paths(self) -> None:
        root = resolve_dcsass_root(self.data_root)
        self.data_root = root
        if self.train_csv is None:
            self.train_csv = root / "splits" / "train.csv"
        else:
            self.train_csv = (self.train_csv if self.train_csv.is_absolute() else (root / self.train_csv)).resolve()
        if self.val_csv is None:
            self.val_csv = root / "splits" / "val.csv"
        else:
            self.val_csv = (self.val_csv if self.val_csv.is_absolute() else (root / self.val_csv)).resolve()
        self.out = self.out.expanduser().resolve()

    @property
    def image_size_tuple(self) -> tuple[int, int]:
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


def _prepare_datasets(config: TrainingConfig):
    entries = load_split_entries(config.data_root, "train", seed=config.seed, csv_path=config.train_csv)

    if config.mil_mode == "oneclass":
        abnormal = [entry for entry in entries if int(entry.get("label_index", 1)) == 1]
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


def run_training(config: TrainingConfig) -> None:
    set_global_seed(config.seed)
    config.resolve_paths()

    config.mil_mode = config.mil_mode.lower()
    if config.mil_mode not in {"oneclass", "posneg"}:
        raise ValueError(f"Unsupported MIL mode '{config.mil_mode}'.")

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    config.out.mkdir(parents=True, exist_ok=True)
    log_dir = config.out / "logs"
    checkpoints_dir = config.out / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch, pos_iter, neg_iter, val_ds = _prepare_datasets(config)

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

        if config.mil_mode == "oneclass":
            assert pos_iter is not None
            for _ in range(steps_per_epoch):
                losses = _train_oneclass_step(encoder, head, optimizer, next(pos_iter), encoder.trainable, config)
                epoch_totals.append(losses["total"])
                epoch_rank.append(losses["ranking"])
                epoch_sparse.append(losses["sparsity"])
                epoch_smooth.append(losses["smoothness"])
                global_step += 1
                with summary_writer.as_default():
                    tf.summary.scalar("train/ocmil_total", losses["total"], step=global_step)
                    tf.summary.scalar("train/ocmil_ranking", losses["ranking"], step=global_step)
                    tf.summary.scalar("train/ocmil_sparsity", losses["sparsity"], step=global_step)
                    tf.summary.scalar("train/ocmil_smoothness", losses["smoothness"], step=global_step)
        else:
            assert pos_iter is not None and neg_iter is not None
            for _ in range(steps_per_epoch):
                losses = _train_posneg_step(encoder, head, optimizer, next(pos_iter), next(neg_iter), encoder.trainable, config)
                epoch_totals.append(losses["total"])
                global_step += 1
                with summary_writer.as_default():
                    tf.summary.scalar("train/total_loss", losses["total"], step=global_step)

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
        config = TrainingConfig.from_yaml(args.config.expanduser().resolve())
    else:
        config = TrainingConfig()
    config.apply_cli(args)
    run_training(config)


if __name__ == "__main__":
    main()
