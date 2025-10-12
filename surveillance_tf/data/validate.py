"""Data validation utilities for split CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from surveillance_tf.data.dcsass_loader import SPLIT_NAMES, load_split_entries
from surveillance_tf.data.transforms import decode_video_opencv
from surveillance_tf.utils.logging import get_logger
from surveillance_tf.utils.paths import resolve_dcsass_root

LOGGER = get_logger(__name__)


def _validate_entry(entry: Dict[str, str], max_frames: int, target_size: Tuple[int, int]) -> List[str]:
    issues: List[str] = []
    video_path = Path(entry["path"])
    if not video_path.exists():
        issues.append("missing")
        return issues
    try:
        size_bytes = video_path.stat().st_size
    except OSError as exc:
        issues.append(f"stat-failed: {exc}")
        return issues
    if size_bytes == 0:
        issues.append("zero-bytes")
    try:
        decode_video_opencv(str(video_path), target_size=target_size, max_frames=max_frames)
    except Exception as exc:  # pragma: no cover - depends on environment/codecs
        issues.append(f"decode-error: {exc}")
    return issues


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=None, help="Dataset root (defaults to ./data/dcsass).")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        choices=SPLIT_NAMES,
        default=list(SPLIT_NAMES),
        help="Splits to validate when CSV paths are not provided.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        help="Explicit CSV file(s) to validate. If provided, --splits is ignored.",
    )
    parser.add_argument("--max_videos", type=int, help="Validate at most N videos per split for a quick check.")
    parser.add_argument("--max_frames", type=int, default=1, help="Number of frames to decode per video during probing.")
    parser.add_argument("--image_size", type=int, nargs=2, default=(224, 224), help="Resize target for probe decoding.")
    parser.add_argument("--seed", type=int, default=1337, help="Seed used for deterministic split loading.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        dataset_root = resolve_dcsass_root(args.data_root)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return

    splits_to_check: List[Tuple[str, List[Dict[str, str]]]] = []

    if args.csv:
        for csv_path in args.csv:
            try:
                entries = load_split_entries(dataset_root, "custom", seed=args.seed, csv_path=csv_path)
            except Exception as exc:
                LOGGER.error("Failed to load CSV %s: %s", csv_path, exc)
                continue
            splits_to_check.append((Path(csv_path).name, entries))
    else:
        for split in args.splits:
            try:
                entries = load_split_entries(dataset_root, split, seed=args.seed)
            except Exception as exc:
                LOGGER.error("Failed to load split '%s': %s", split, exc)
                continue
            splits_to_check.append((split, entries))

    total_errors = 0
    total_warnings = 0
    target_size = (int(args.image_size[0]), int(args.image_size[1]))

    for split_name, entries in splits_to_check:
        if not entries:
            LOGGER.warning("No entries found for split '%s'.", split_name)
            continue
        LOGGER.info("Validating split '%s' (%d entries)...", split_name, len(entries))
        to_iterate = entries[: args.max_videos] if args.max_videos else entries
        for entry in to_iterate:
            issues = _validate_entry(entry, max_frames=args.max_frames, target_size=target_size)
            if issues:
                issue_list = ", ".join(issues)
                LOGGER.error("  [%s] %s -> %s", split_name, entry["path"], issue_list)
                total_errors += 1 if "missing" in issues or "decode-error" in issue_list else 0
                total_warnings += 1 if "zero-bytes" in issues else 0

    if total_errors == 0 and total_warnings == 0:
        LOGGER.info("Data validation complete: no issues detected.")
    else:
        LOGGER.info(
            "Data validation finished with %d error(s) and %d warning(s).",
            total_errors,
            total_warnings,
        )


if __name__ == "__main__":
    main()


__all__ = ["main", "parse_args"]
