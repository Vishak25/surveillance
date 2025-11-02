"""Video decoding and segmentation helpers."""
from __future__ import annotations

import os
from typing import List, Sequence

import cv2
import imageio
import numpy as np

from ..utils.logging import get_logger

# Set ffmpeg path for imageio - check Homebrew locations first
if "IMAGEIO_FFMPEG_EXE" not in os.environ:
    if os.path.exists("/opt/homebrew/bin/ffmpeg"):
        os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
    elif os.path.exists("/usr/local/bin/ffmpeg"):
        os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/local/bin/ffmpeg"
    elif os.path.exists("/usr/bin/ffmpeg"):
        os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

LOGGER = get_logger(__name__)


def decode_video_opencv(
    path: str,
    target_size: Sequence[int] = (224, 224),
    max_frames: int | None = None,
) -> List[np.ndarray]:
    """Decode frames with imageio and resize via OpenCV.

    Parameters
    ----------
    path:
        Path to the video file.
    target_size:
        Output spatial size as (height, width).
    max_frames:
        Optional cap on the number of frames to decode. Useful for quick probes/validation.
    """
    try:
        reader = imageio.get_reader(path, "ffmpeg")
    except Exception as exc:  # pragma: no cover - dependent on external codecs
        raise RuntimeError(f"Failed to open video {path}") from exc

    target_h, target_w = int(target_size[0]), int(target_size[1])
    frames: List[np.ndarray] = []
    try:
        for idx, frame in enumerate(reader):
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            frames.append(resized)
            if max_frames is not None and idx + 1 >= max_frames:
                break
    except Exception as exc:  # pragma: no cover - decode errors are environment specific
        raise RuntimeError(f"Error decoding video {path}: {exc}") from exc
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    LOGGER.debug("Decoded %d frames from %s", len(frames), path)
    return frames


def make_segments(frames: Sequence[np.ndarray], T: int, stride: int) -> np.ndarray:
    """Create sliding-window segments of length ``T`` from a sequence of frames."""
    if not frames:
        return np.empty((0, T, 224, 224, 3), dtype=np.uint8)
    arr = np.stack(frames)
    windows = []
    for start in range(0, len(frames) - T + 1, stride):
        windows.append(arr[start : start + T])
    if not windows:
        return np.empty((0, T) + arr.shape[1:], dtype=arr.dtype)
    stacked = np.stack(windows)
    return stacked


__all__ = ["decode_video_opencv", "make_segments"]
