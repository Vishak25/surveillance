"""DCSASS data loader with optional metadata support."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from surveillance_tf.utils.logging import get_logger
from surveillance_tf.utils.paths import list_videos, resolve_dcsass_root
from surveillance_tf.utils.seed import set_global_seed
from surveillance_tf.data.transforms import decode_video_opencv, make_segments

LOGGER = get_logger(__name__)

VIDEO_EXTS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
SPLIT_NAMES = ("train", "val", "test")

METADATA_PATH_CANDIDATES = ("path", "video", "filepath", "file", "relative_path")
METADATA_LABEL_CANDIDATES = ("label", "class", "category", "target", "y")
METADATA_SPLIT_CANDIDATES = ("split", "partition", "set")


def _read_metadata_csv(metadata_path: Path):
    import pandas as pd

    df = pd.read_csv(metadata_path)
    if df.empty:
        raise ValueError(f"Metadata file {metadata_path} is empty.")
    return df


def _detect_column(df, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    return None


def _normalize_video_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path

    candidate = (root / path).resolve()
    if candidate.exists():
        return candidate

    dataset_dir = root / "DCSASS Dataset"
    if dataset_dir.exists():
        candidate = (dataset_dir / path).resolve()
        if candidate.exists():
            return candidate

    matches = list(root.rglob(path.name))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Unable to resolve video path '{raw_path}' under {root}")


def _assign_splits(entries: List[Dict[str, str]], seed: int) -> None:
    labels = [entry["label"] for entry in entries]
    idx = np.arange(len(entries))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(sss.split(idx, labels))

    def set_split(indices: Iterable[int], name: str) -> None:
        for i in indices:
            entries[i]["split"] = name

    set_split(train_idx, "train")

    temp_labels = [labels[i] for i in temp_idx]
    temp_idx_array = np.array(temp_idx)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(sss_val.split(temp_idx_array, temp_labels))
    set_split(temp_idx_array[val_idx], "val")
    set_split(temp_idx_array[test_idx], "test")


def _write_inferred_splits(root: Path, entries: List[Dict[str, str]]) -> None:
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        csv_path = splits_dir / f"{split}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["video", "label"])
            writer.writerows(
                (Path(entry["path"]).resolve().relative_to(root), entry["label"])
                for entry in entries
                if entry.get("split") == split
            )
        LOGGER.info(
            "Wrote inferred %s split with %d entries to %s",
            split,
            sum(1 for e in entries if e.get("split") == split),
            csv_path,
        )


def _load_metadata(root: Path, seed: int = 1337) -> List[Dict[str, str]]:
    metadata_path = root / "metadata.csv"
    entries: List[Dict[str, str]] = []

    if metadata_path.exists():
        df = _read_metadata_csv(metadata_path)
        path_col = _detect_column(df, METADATA_PATH_CANDIDATES)
        label_col = _detect_column(df, METADATA_LABEL_CANDIDATES)

        if path_col is None or label_col is None:
            raise RuntimeError(
                "metadata.csv must contain video path and label columns. "
                f"Checked for {METADATA_PATH_CANDIDATES} and {METADATA_LABEL_CANDIDATES}."
            )

        split_col = _detect_column(df, METADATA_SPLIT_CANDIDATES)
        if split_col is None:
            _assign_splits_from_dataframe(df, label_col, seed=seed)
        else:
            df["split"] = df[split_col].astype(str)

        for _, row in df.iterrows():
            video_path = _normalize_video_path(root, str(row[path_col]))
            entries.append(
                {
                    "path": str(video_path),
                    "label": str(row[label_col]),
                    "split": str(row.get("split", "train")).lower(),
                }
            )
    else:
        videos = list_videos(root, VIDEO_EXTS)
        if not videos:
            raise FileNotFoundError(
                f"No video files found under {root}. Ensure the dataset is extracted correctly."
            )
        for video in videos:
            entries.append({"path": str(video), "label": video.parent.name})
        _assign_splits(entries, seed)
        _write_inferred_splits(root, entries)

    has_normal = any(entry["label"].lower() == "normal" for entry in entries)
    for entry in entries:
        is_normal = entry["label"].lower() == "normal"
        entry["binary_label"] = 0 if is_normal else 1
        if has_normal and is_normal:
            entry["label_index"] = 0
        else:
            entry["label_index"] = 1
    return entries


def _assign_splits_from_dataframe(df, label_col: str, seed: int) -> None:
    labels = df[label_col].astype(str).tolist()
    idx = np.arange(len(labels))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(sss.split(idx, labels))
    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    temp_labels = [labels[i] for i in temp_idx]
    temp_idx_array = np.array(temp_idx)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(sss_val.split(temp_idx_array, temp_labels))
    split[temp_idx_array[val_idx]] = "val"
    split[temp_idx_array[test_idx]] = "test"
    df["split"] = split


def _select_entries(entries: List[Dict[str, str]], selected_paths: List[Path], split: str) -> List[Dict[str, str]]:
    path_map: Dict[Path, Dict[str, str]] = {}
    for entry in entries:
        full_path = Path(entry["path"]).resolve()
        path_map[full_path] = entry

    selected: List[Dict[str, str]] = []
    for path in selected_paths:
        entry = path_map.get(path.resolve())
        if entry is None:
            label = path.parent.name
            is_normal = label.lower() == "normal"
            selected.append(
                {
                    "path": str(path),
                    "label": label,
                    "label_index": 0 if is_normal else 1,
                    "binary_label": 0 if is_normal else 1,
                    "split": split,
                }
            )
        else:
            clone = entry.copy()
            clone["split"] = split
            selected.append(clone)
    if selected:
        unresolved = [entry for entry in selected if entry["label_index"] == -1]
        if unresolved:
            existing_labels = {entry["label"]: entry["label_index"] for entry in entries}
            next_index = max(existing_labels.values(), default=-1) + 1
            for entry in selected:
                if entry["label_index"] == -1:
                    label = entry["label"]
                    entry["label_index"] = existing_labels.setdefault(label, next_index)
                    if entry["label_index"] == next_index:
                        next_index += 1
    return selected


def load_split_entries(
    root: str | Path,
    split: str,
    seed: int = 1337,
    csv_path: Optional[str | Path] = None,
) -> List[Dict[str, str]]:
    dataset_root = resolve_dcsass_root(root)
    entries = _load_metadata(dataset_root, seed=seed)
    if csv_path is not None:
        csv_file = Path(csv_path)
        if not csv_file.is_absolute():
            csv_file = dataset_root / csv_file
        if not csv_file.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_file}")
        selected_paths: List[Path] = []
        with csv_file.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "video" not in reader.fieldnames:
                raise ValueError("CSV must contain 'video' column")
            for row in reader:
                video_path = _normalize_video_path(dataset_root, row["video"])
                selected_paths.append(video_path)
        filtered = _select_entries(entries, selected_paths, split)
    else:
        filtered = [entry.copy() for entry in entries if entry.get("split") == split]
    if not filtered:
        raise ValueError(f"No entries found for split '{split}'.")
    return filtered


def make_bag_dataset(
    root: str | Path,
    split: str,
    T: int = 32,
    stride: int = 3,
    batch_size: int = 1,
    image_size: Tuple[int, int] = (224, 224),
    seed: int = 1337,
    csv_path: Optional[str | Path] = None,
    entries: Optional[Sequence[Dict[str, str]]] = None,
) -> "tf.data.Dataset":


    dataset_root = resolve_dcsass_root(root)
    if entries is not None:
        entries_list = [entry.copy() for entry in entries]
    else:
        entries_list = load_split_entries(dataset_root, split, seed=seed, csv_path=csv_path)

    def generator() -> Iterator[Tuple[np.ndarray, np.int32, bytes]]:
        for entry in entries_list:
            video_path = Path(entry["path"])
            try:
                frames = decode_video_opencv(str(video_path), target_size=image_size)
            except FileNotFoundError:
                LOGGER.warning("Skipping missing video %s", video_path)
                continue
            segments = make_segments(frames, T=T, stride=stride)
            if segments.size == 0:
                continue
            segments = segments.astype(np.float32) / 255.0
            yield segments, np.int32(entry["binary_label"]), str(video_path).encode("utf-8")

    output_signature = (
        tf.TensorSpec(shape=(None, T, image_size[0], image_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.string),
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.map(
        lambda seg, label, vid: (tf.RaggedTensor.from_tensor(seg), label, vid),
        num_parallel_calls=1,
    )
    ds = ds.batch(batch_size).prefetch(1)
    return ds


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument("--split", choices=SPLIT_NAMES, default="train")
    parser.add_argument("--csv", type=Path, help="Optional CSV specifying video,label for the split")
    parser.add_argument("--make_splits", type=Path, help="Write inferred splits CSVs to this directory")
    parser.add_argument("--sample", action="store_true", help="Decode and report the first video in the split.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--image_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--stride", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_global_seed(args.seed)
    try:
        dataset_root = resolve_dcsass_root(args.data_root)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return

    entries = _load_metadata(dataset_root, seed=args.seed)
    LOGGER.info("Discovered %d videos across dataset.", len(entries))

    if args.make_splits:
        target = args.make_splits
        target.mkdir(parents=True, exist_ok=True)
        for split_name in SPLIT_NAMES:
            filtered = [entry for entry in entries if entry.get("split") == split_name]
            csv_path = target / f"{split_name}.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["video", "label"])
                writer.writerows((Path(entry["path"]).resolve().relative_to(dataset_root), entry["label"]) for entry in filtered)
            LOGGER.info("Wrote %s with %d entries", csv_path, len(filtered))

    if args.sample:
        try:
            import tensorflow as tf

            ds = make_bag_dataset(
                dataset_root,
                split=args.split,
                T=args.T,
                stride=args.stride,
                batch_size=1,
                image_size=tuple(args.image_size),
                seed=args.seed,
                csv_path=args.csv,
            )
            first = next(iter(ds))
        except StopIteration:
            LOGGER.warning("No samples available in split '%s'.", args.split)
            return
        segments, label, video_id = first
        LOGGER.info(
            "Sampled video %s -> ragged segments shape %s, label=%s",
            video_id.numpy()[0].decode("utf-8"),
            segments.shape,
            label.numpy()[0],
        )


if __name__ == "__main__":
    main()


__all__ = ["make_bag_dataset", "load_split_entries"]
