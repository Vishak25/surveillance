"""Utilities for resolving dataset locations and listing video files."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .logging import get_logger

LOGGER = get_logger(__name__)

VIDEO_EXTS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")

EXPECTED_LAYOUT = """\
data/
├── dcsass/
│   ├── DCSASS Dataset/
│   ├── sample/
│   └── metadata.csv (optional)
"""


def _normalize(path: Path) -> Path:
    return path.expanduser().resolve()


def _candidate_roots(explicit_root: Optional[Path]) -> List[Path]:
    candidates: List[Path] = []
    if explicit_root is not None:
        candidates.append(_normalize(explicit_root))

    env_var = os.getenv("DCSASS_DATA_DIR")
    if env_var:
        candidates.append(_normalize(Path(env_var)))

    candidates.append(_normalize(Path("data/dcsass")))

    kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets"
    if kaggle_cache.exists():
        for owner in kaggle_cache.iterdir():
            version_root = owner / "dcsass-dataset" / "versions"
            if version_root.exists():
                for version in sorted(version_root.glob("*"), reverse=True):
                    ds_dir = version / "DCSASS Dataset"
                    if ds_dir.exists():
                        candidates.append(_normalize(ds_dir.parent))

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def resolve_dcsass_root(explicit_root: Optional[str | Path]) -> Path:
    """Resolve the DCSASS dataset root directory, with informative errors if missing."""
    explicit_path = Path(explicit_root) if explicit_root is not None else None
    candidates = _candidate_roots(explicit_path)
    for candidate in candidates:
        if candidate.exists():
            LOGGER.debug("Resolved DCSASS dataset root at %s", candidate)
            return candidate

    message = (
        "Could not find the 'dcsass' dataset.\n"
        f"Tried the following locations:\n"
        + "\n".join(f"  - {path}" for path in candidates)
        + "\n\nExpected directory layout:\n"
        + EXPECTED_LAYOUT
        + "\nProvide the correct path via '--data_root PATH' or set the environment variable 'DCSASS_DATA_DIR'."
    )
    raise FileNotFoundError(message)


def list_videos(root: Path, exts: Sequence[str] = VIDEO_EXTS) -> List[Path]:
    """Return sorted list of video files recursively under root."""
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    files.sort()
    return files


def find_video_dirs(root: Path) -> List[Path]:
    """Return directories containing video files under the dataset root."""
    dirs: List[Path] = []
    dataset_root = _normalize(root)
    dcsass_main = dataset_root / "DCSASS Dataset"
    sample_dir = dataset_root / "sample"

    def _collect(base: Path) -> None:
        if base.exists():
            for video in list_videos(base):
                parent = video.parent
                if parent not in dirs:
                    dirs.append(parent)

    if dcsass_main.exists():
        _collect(dcsass_main)
        if sample_dir.exists():
            _collect(sample_dir)
    else:
        _collect(dataset_root)
    return dirs


__all__ = ["resolve_dcsass_root", "find_video_dirs", "list_videos", "VIDEO_EXTS"]
