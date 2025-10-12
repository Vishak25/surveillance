"""MIL training script (DCSASS)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

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

try:
    from surveillance_tf.data import ucfcrime_loader  # type: ignore
except ImportError:  # pragma: no cover - optional dataset support
    ucfcrime_loader = None


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config_yaml",
        type=Path,
        help="Path to YAML experiment config providing hyperparameters/defaults.",
    )
    config_args, remaining = config_parser.parse_known_args(list(argv) if argv is not None else None)

    config_defaults: dict = {}
    if config_args.config_yaml:
        config_path = config_args.config_yaml.expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config YAML not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if loaded and not isinstance(loaded, dict):
            raise ValueError(f"Config YAML must contain a mapping at the root: {config_path}")
        config_defaults = loaded or {}

    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[config_parser],
    )
    parser.add_argument("--dataset", choices=("dcsass", "ucfcrime"), default="dcsass")
    parser.add_argument("--data_root", type=Path, default=None, help="Dataset root (defaults to ./data/dcsass).")
    parser.add_argument("--train_csv", type=Path, help="Optional CSV overriding the training split.")
    parser.add_argument("--val_csv", type=Path, help="Optional CSV overriding the validation split.")
    parser.add_argument("--out", type=Path, default=None, help="Training output directory.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mil_mode", choices=("oneclass", "posneg"), default=None)
    parser.add_argument("--k", type=int, default=3, help="Top/Bottom-k for One-Class MIL ranking loss.")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lambda_sparse", type=float, default=8e-5)
    parser.add_argument("--lambda_smooth", type=float, default=0.1)
    parser.add_argument("--freeze_backbone_until", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument(
        "--experiment_tracker",
        choices=("none", "wandb"),
        default="none",
        help="Optional experiment tracker integration.",
    )
    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, help="Weights & Biases entity (team) name.")
    parser.add_argument("--wandb_dir", type=Path, help="Weights & Biases run directory override.")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="Optional W&B tags.")
    parser.add_argument("--run_name", type=str, help="Human-readable experiment/run identifier.")

    valid_dests = {action.dest for action in parser._actions if action.dest != "help"}
    applicable_defaults = {k: v for k, v in config_defaults.items() if k in valid_dests}
    unused_keys = sorted(set(config_defaults.keys()) - set(applicable_defaults.keys()))
    if applicable_defaults:
        parser.set_defaults(**applicable_defaults)

    args = parser.parse_args(list(argv) if argv is not None else None)
    args.loaded_config = config_defaults
    args.unused_config_keys = unused_keys
    return args


def _ensure_positive_negative(entries: List[dict]) -> None:
    labels = [int(entry["binary_label"]) for entry in entries]
    n_pos = sum(1 for label in labels if label == 1)
    n_neg = sum(1 for label in labels if label == 0)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training split must contain both positive and negative videos.")


def _default_split(root: Path, name: str) -> Path:
    return root / "splits" / f"{name}.csv"


def _resolve_dataset_root(dataset: str, data_root: Optional[Path]) -> Path:
    if dataset == "dcsass":
        return resolve_dcsass_root(data_root)
    if dataset == "ucfcrime":
        if ucfcrime_loader is None:
            raise RuntimeError(
                "UCF-Crime dataset selected but 'surveillance_tf.data.ucfcrime_loader' is unavailable."
                " Restore the loader module or reinstall optional dependencies."
            )
        if hasattr(ucfcrime_loader, "resolve_ucfcrime_root"):
            return ucfcrime_loader.resolve_ucfcrime_root(data_root)  # type: ignore[attr-defined]
        if data_root is None:
            candidate = Path("data/ucf_crime")
        else:
            candidate = data_root
        root = candidate.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found at {root}")
        return root
    raise ValueError(f"Unsupported dataset '{dataset}'.")


def _select_loaders(dataset: str):
    if dataset == "dcsass":
        return load_split_entries, make_bag_dataset
    if dataset == "ucfcrime":
        if ucfcrime_loader is None:
            raise RuntimeError(
                "UCF-Crime dataset selected but loader utilities are unavailable."
            )
        return ucfcrime_loader.load_split_entries, ucfcrime_loader.make_bag_dataset  # type: ignore[attr-defined]
    raise ValueError(f"Unsupported dataset '{dataset}'.")


def _coerce_arg_types(args: argparse.Namespace) -> None:
    path_fields = ("data_root", "train_csv", "val_csv", "out", "wandb_dir", "config_yaml")
    for field in path_fields:
        value = getattr(args, field, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, field, Path(value))

    int_fields = ("epochs", "k", "freeze_backbone_until", "seed", "batch_size", "T", "stride")
    for field in int_fields:
        value = getattr(args, field, None)
        if value is not None and not isinstance(value, int):
            setattr(args, field, int(value))

    float_fields = ("lr", "margin", "lambda_sparse", "lambda_smooth")
    for field in float_fields:
        value = getattr(args, field, None)
        if value is not None and not isinstance(value, float):
            setattr(args, field, float(value))

    if isinstance(args.image_size, (list, tuple)):
        args.image_size = tuple(int(v) for v in args.image_size)
    else:
        args.image_size = (224, 224)


def _build_run_config(args: argparse.Namespace) -> dict:
    config = {
        "dataset": args.dataset,
        "data_root": str(args.data_root) if args.data_root else None,
        "train_csv": str(args.train_csv) if args.train_csv else None,
        "val_csv": str(args.val_csv) if args.val_csv else None,
        "out": str(args.out) if args.out else None,
        "epochs": args.epochs,
        "lr": args.lr,
        "mil_mode": args.mil_mode,
        "k": args.k,
        "margin": args.margin,
        "lambda_sparse": args.lambda_sparse,
        "lambda_smooth": args.lambda_smooth,
        "freeze_backbone_until": args.freeze_backbone_until,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "image_size": list(args.image_size),
        "T": args.T,
        "stride": args.stride,
        "experiment_tracker": args.experiment_tracker,
        "run_name": args.run_name,
    }
    if args.config_yaml:
        config["config_yaml"] = str(args.config_yaml)
    if getattr(args, "loaded_config", None):
        config["config_defaults"] = args.loaded_config
    return config


def _init_experiment_tracker(args: argparse.Namespace, run_config: dict):
    if args.experiment_tracker == "none":
        return None, None
    if args.experiment_tracker == "wandb":
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - optional dep
            LOGGER.error("Weights & Biases requested but package not installed: %s", exc)
            return None, None
        if not args.wandb_project:
            LOGGER.error("Weights & Biases enabled but --wandb_project missing.")
            return None, None
        init_kwargs = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.run_name,
            "dir": str(args.wandb_dir) if args.wandb_dir else None,
            "tags": args.wandb_tags,
            "config": run_config,
        }
        LOGGER.info("Initialising Weights & Biases run (project=%s, name=%s)", args.wandb_project, args.run_name)
        run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})
        return run, wandb
    LOGGER.warning("Unsupported experiment tracker '%s'; continuing without tracking.", args.experiment_tracker)
    return None, None


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    _coerce_arg_types(args)

    if args.unused_config_keys:
        LOGGER.warning(
            "Ignored config keys without corresponding CLI arguments: %s",
            ", ".join(args.unused_config_keys),
        )

    if args.out is None:
        raise ValueError("Output directory must be specified via --out or the config YAML.")
    args.out = args.out.expanduser().resolve()
    if args.config_yaml:
        args.config_yaml = args.config_yaml.expanduser().resolve()

    set_global_seed(args.seed)

    if args.mil_mode is None:
        args.mil_mode = "oneclass" if args.dataset == "dcsass" else "posneg"

    run_config = _build_run_config(args)
    tracker_run, tracker_module = _init_experiment_tracker(args, run_config)
    if tracker_run is not None and args.config_yaml and tracker_module is not None:
        config_artifact = tracker_module.Artifact("training-config", type="config")
        config_artifact.add_file(str(args.config_yaml))
        tracker_run.log_artifact(config_artifact)

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    if args.mil_mode == "oneclass" and args.dataset != "dcsass":
        LOGGER.error("One-Class MIL is currently supported only for the DCSASS dataset.")
        return

    try:
        dataset_root = _resolve_dataset_root(args.dataset, args.data_root)
        load_entries_fn, make_dataset_fn = _select_loaders(args.dataset)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error(str(exc))
        return

    train_csv = args.train_csv or _default_split(dataset_root, "train")
    val_csv = args.val_csv or _default_split(dataset_root, "val")

    all_train_entries = load_entries_fn(dataset_root, "train", seed=args.seed, csv_path=train_csv)
    if args.mil_mode == "oneclass":
        abnormal_train_entries = [entry for entry in all_train_entries if int(entry.get("label_index", 1)) == 1]
        if not abnormal_train_entries:
            LOGGER.error("No abnormal videos found in the training split for One-Class MIL.")
            return
        LOGGER.info("Starting One-Class MIL training with %d abnormal videos.", len(abnormal_train_entries))
        train_ds = make_dataset_fn(
            dataset_root,
            split="train",
            entries=abnormal_train_entries,
            T=args.T,
            stride=args.stride,
            batch_size=1,
            image_size=tuple(args.image_size),
            seed=args.seed,
        )
        train_ds = train_ds.map(lambda segments, label, vid: segments, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        train_iter = iter(train_ds.repeat())
        steps_per_epoch = len(abnormal_train_entries)
    else:
        _ensure_positive_negative(all_train_entries)
        train_ds = make_dataset_fn(
            dataset_root,
            split="train",
            csv_path=train_csv,
            T=args.T,
            stride=args.stride,
            batch_size=args.batch_size,
            image_size=tuple(args.image_size),
            seed=args.seed,
        )
        train_ds = train_ds.map(lambda seg, label, vid: (seg, label, vid), num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        n_pos = sum(1 for entry in all_train_entries if int(entry["binary_label"]) == 1)
        n_neg = sum(1 for entry in all_train_entries if int(entry["binary_label"]) == 0)
        pos_ds = train_ds.filter(lambda seg, label, vid: tf.equal(label, 1))
        neg_ds = train_ds.filter(lambda seg, label, vid: tf.equal(label, 0))
        pos_iter = iter(pos_ds.repeat())
        neg_iter = iter(neg_ds.repeat())
        steps_per_epoch = min(n_pos, n_neg)

    if tracker_run is not None:
        tracker_run.log(
            {
                "data/train_entries": len(all_train_entries),
                "data/steps_per_epoch": steps_per_epoch,
            },
            step=0,
        )

    val_ds = make_dataset_fn(
        dataset_root,
        split="val",
        csv_path=val_csv,
        T=args.T,
        stride=args.stride,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
        seed=args.seed,
    )
    val_ds = val_ds.map(lambda seg, label, vid: (seg, label, vid), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    encoder = build_segment_encoder(trainable=True)
    head = MILScoringHead()
    optimizer = tf.keras.optimizers.Adam(args.lr)

    out_dir = args.out
    log_dir = out_dir / "logs"
    model_dir = out_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    writer = tf.summary.create_file_writer(str(log_dir))
    best_auc = 0.0
    best_ckpt_path = model_dir / "ckpt_best"
    global_step = 0

    def log_tracker(metrics: dict, step: int | None = None) -> None:
        if tracker_run is not None:
            tracker_run.log(metrics, step=step)

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

    @tf.function
    def train_step_oneclass(abnormal_bag, train_backbone: tf.Tensor):
        segments = abnormal_bag
        if isinstance(segments, tf.RaggedTensor):
            segments_dense = segments.to_tensor()
        else:
            segments_dense = tf.convert_to_tensor(segments)
        segments_dense = _maybe_squeeze_first_dim(segments_dense)
        train_flag = tf.cast(train_backbone, tf.bool)
        with tf.GradientTape() as tape:
            features = encoder(segments_dense, training=train_flag)
            raw_scores = tf.squeeze(head(features, training=True), axis=-1)
            raw_scores = _ensure_batch_dim(raw_scores)
            ragged_scores = tf.RaggedTensor.from_tensor(raw_scores)
            losses = compute_one_class_mil_losses(
                bag_scores=ragged_scores,
                k=args.k,
                margin=args.margin,
                lambda_sparse=args.lambda_sparse,
                lambda_smooth=args.lambda_smooth,
            )
            total_loss = losses["total"]
        variables = encoder.trainable_variables + head.trainable_variables
        grads = tape.gradient(total_loss, variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, variables) if g is not None]
        if grads_and_vars:
            optimizer.apply_gradients(grads_and_vars)
        return losses

    @tf.function
    def train_step_posneg(pos_segments, neg_segments, train_backbone: tf.Tensor):
        pos_dense = pos_segments.to_tensor() if isinstance(pos_segments, tf.RaggedTensor) else tf.convert_to_tensor(pos_segments)
        neg_dense = neg_segments.to_tensor() if isinstance(neg_segments, tf.RaggedTensor) else tf.convert_to_tensor(neg_segments)
        pos_dense = _maybe_squeeze_first_dim(pos_dense)
        neg_dense = _maybe_squeeze_first_dim(neg_dense)
        train_flag = tf.cast(train_backbone, tf.bool)
        with tf.GradientTape() as tape:
            pos_features = encoder(pos_dense, training=train_flag)
            neg_features = encoder(neg_dense, training=train_flag)
            pos_scores = tf.squeeze(head(pos_features, training=True), axis=-1)
            neg_scores = tf.squeeze(head(neg_features, training=True), axis=-1)
            pos_scores = _ensure_batch_dim(pos_scores)
            neg_scores = _ensure_batch_dim(neg_scores)
            pos_rt = tf.RaggedTensor.from_tensor(pos_scores)
            neg_rt = tf.RaggedTensor.from_tensor(neg_scores)
            losses = compute_losses(
                pos_rt,
                neg_rt,
                margin=args.margin,
                lambda_sparse=args.lambda_sparse,
                lambda_smooth=args.lambda_smooth,
            )
            total_loss = losses["total"]
        variables = encoder.trainable_variables + head.trainable_variables
        grads = tape.gradient(total_loss, variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, variables) if g is not None]
        if grads_and_vars:
            optimizer.apply_gradients(grads_and_vars)
        return losses

    for epoch in range(1, args.epochs + 1):
        encoder.trainable = epoch > args.freeze_backbone_until
        epoch_totals: List[float] = []
        epoch_rankings: List[float] = []
        epoch_sparsities: List[float] = []
        epoch_smoothness: List[float] = []

        if args.mil_mode == "oneclass":
            for _ in range(steps_per_epoch):
                losses = train_step_oneclass(next(train_iter), tf.constant(encoder.trainable))
                total = float(losses["total"].numpy())
                ranking = float(losses["ranking"].numpy())
                sparsity = float(losses["sparsity"].numpy())
                smoothness = float(losses["smoothness"].numpy())
                epoch_totals.append(total)
                epoch_rankings.append(ranking)
                epoch_sparsities.append(sparsity)
                epoch_smoothness.append(smoothness)
                global_step += 1
                with writer.as_default():
                    tf.summary.scalar("ocmil/ranking", ranking, step=global_step)
                    tf.summary.scalar("ocmil/sparsity", sparsity, step=global_step)
                    tf.summary.scalar("ocmil/smoothness", smoothness, step=global_step)
                    tf.summary.scalar("ocmil/total", total, step=global_step)
                log_tracker(
                    {
                        "train/ocmil_ranking": ranking,
                        "train/ocmil_sparsity": sparsity,
                        "train/ocmil_smoothness": smoothness,
                        "train/ocmil_total": total,
                    },
                    step=global_step,
                )
        else:
            for _ in range(steps_per_epoch):
                pos_batch = next(pos_iter)
                neg_batch = next(neg_iter)
                losses = train_step_posneg(
                    pos_batch[0],
                    neg_batch[0],
                    tf.constant(encoder.trainable),
                )
                total = float(losses["total"].numpy())
                epoch_totals.append(total)
                global_step += 1
                with writer.as_default():
                    tf.summary.scalar("train/total_loss", total, step=global_step)
                log_tracker({"train/total_loss": total}, step=global_step)

        mean_total = float(np.mean(epoch_totals)) if epoch_totals else float("nan")
        LOGGER.info("Epoch %d/%d [%s]: mean total loss %.5f", epoch, args.epochs, args.mil_mode, mean_total)
        log_tracker({"train/epoch_total_loss": mean_total}, step=epoch)

        if args.mil_mode == "oneclass":
            mean_ranking = float(np.mean(epoch_rankings)) if epoch_rankings else float("nan")
            mean_sparsity = float(np.mean(epoch_sparsities)) if epoch_sparsities else float("nan")
            mean_smooth = float(np.mean(epoch_smoothness)) if epoch_smoothness else float("nan")
            with writer.as_default():
                tf.summary.scalar("ocmil/epoch_ranking", mean_ranking, step=epoch)
                tf.summary.scalar("ocmil/epoch_sparsity", mean_sparsity, step=epoch)
                tf.summary.scalar("ocmil/epoch_smoothness", mean_smooth, step=epoch)
                tf.summary.scalar("ocmil/epoch_total", mean_total, step=epoch)
            log_tracker(
                {
                    "train/epoch_ranking": mean_ranking,
                    "train/epoch_sparsity": mean_sparsity,
                    "train/epoch_smoothness": mean_smooth,
                },
                step=epoch,
            )

        val_labels: List[int] = []
        val_scores: List[float] = []
        for segments, label, _ in val_ds:
            seg_tensor = segments.to_tensor() if isinstance(segments, tf.RaggedTensor) else tf.convert_to_tensor(segments)
            seg_tensor = _maybe_squeeze_first_dim(seg_tensor)
            features = encoder(seg_tensor, training=False)
            scores = tf.squeeze(head(features, training=False), axis=-1)
            score = float(tf.reduce_max(scores).numpy()) if tf.size(scores) > 0 else 0.0
            val_labels.append(int(label.numpy()))
            val_scores.append(score)

        labels_arr = np.array(val_labels)
        scores_arr = np.array(val_scores)
        unique_labels = np.unique(labels_arr)
        val_auc: Optional[float] = None
        with writer.as_default():
            tf.summary.scalar("train_loss", mean_total, step=epoch)
        log_tracker({"train/loss": mean_total}, step=epoch)

        if unique_labels.size >= 2:
            val_auc = roc_auc(labels_arr, scores_arr)
            with writer.as_default():
                tf.summary.scalar("val_auc", val_auc, step=epoch)
            log_tracker({"val/auc": val_auc}, step=epoch)
        else:
            LOGGER.warning(
                "Validation split contains only one class (label=%s); skipping ROC-AUC computation.",
                unique_labels[0] if unique_labels.size == 1 else "unknown",
            )

        log_tracker(
            {
                "val/score_mean": float(np.mean(scores_arr)) if scores_arr.size else float("nan"),
                "val/score_std": float(np.std(scores_arr)) if scores_arr.size else float("nan"),
            },
            step=epoch,
        )

        if val_auc is not None and val_auc > best_auc:
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
            log_tracker({"val/best_auc": best_auc}, step=epoch)
            if tracker_run is not None and tracker_module is not None:
                artifact = tracker_module.Artifact(f"mil-best-epoch{epoch}", type="model")
                artifact.add_dir(str(best_ckpt_path))
                tracker_run.log_artifact(artifact)

    LOGGER.info(
        "Training complete. Best AUC %.4f. Saved checkpoint to %s",
        best_auc,
        best_ckpt_path,
    )
    if tracker_run is not None:
        tracker_run.summary["best_val_auc"] = best_auc
        tracker_run.finish()


if __name__ == "__main__":
    main()
