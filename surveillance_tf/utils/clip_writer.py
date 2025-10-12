"""Clip writing utilities."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

from .logging import get_logger

LOGGER = get_logger(__name__)

FrameWithTs = Tuple[np.ndarray, float]


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def write_clip_from_frames(frames_with_ts: Iterable[FrameWithTs], fps: float, out_mp4_path: str | Path) -> Path:
    """Write frames to an MP4 clip via ffmpeg or cv2 fallback."""
    frames = list(frames_with_ts)
    if not frames:
        raise ValueError("No frames provided")
    out_path = Path(out_mp4_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _ffmpeg_available():
        LOGGER.debug("Writing clip via ffmpeg")
        temp = out_path.with_suffix(".rawavi")
        height, width = frames[0][0].shape[:2]
        writer = cv2.VideoWriter(
            str(temp),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        for frame, _ in frames:
            writer.write(frame)
        writer.release()
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        temp.unlink(missing_ok=True)
    else:
        LOGGER.warning("ffmpeg not found, using OpenCV VideoWriter")
        height, width = frames[0][0].shape[:2]
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for frame, _ in frames:
            writer.write(frame)
        writer.release()

    LOGGER.info("Clip saved to %s (%d frames)", out_path, len(frames))
    return out_path


__all__ = ["write_clip_from_frames"]
