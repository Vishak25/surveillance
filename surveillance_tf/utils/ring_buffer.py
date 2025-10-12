"""Video ring buffer to retain recent frames."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Tuple

import cv2
import numpy as np

from .logging import get_logger

LOGGER = get_logger(__name__)

FrameRecord = Tuple[np.ndarray, float]


@dataclass
class VideoRingBuffer:
    """Fixed-size buffer storing frames and timestamps."""

    seconds: int
    fps: float

    def __post_init__(self) -> None:
        self.capacity = max(int(self.seconds * self.fps), 1)
        self._frames: Deque[FrameRecord] = deque(maxlen=self.capacity)
        LOGGER.debug("VideoRingBuffer capacity=%d", self.capacity)

    def push(self, frame: np.ndarray, timestamp: float) -> None:
        self._frames.append((frame.copy(), float(timestamp)))

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._frames)

    def iter_window(self, t0: float, t1: float) -> Iterable[FrameRecord]:
        return (rec for rec in self._frames if t0 <= rec[1] <= t1)

    def export_clip(self, t0: float, t1: float, out_path: str | Path, fps: float | None = None) -> Path:
        """Export frames between timestamps [t0, t1] to an MP4 file."""
        frames = list(self.iter_window(t0, t1))
        if not frames:
            raise ValueError("No frames available in requested window")
        fps = fps or self.fps
        height, width = frames[0][0].shape[:2]
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for frame, _ in frames:
            writer.write(frame)
        writer.release()
        LOGGER.info("Exported clip with %d frames to %s", len(frames), out_path)
        return out_path

    @property
    def frames(self) -> List[FrameRecord]:  # pragma: no cover - convenience
        return list(self._frames)


__all__ = ["VideoRingBuffer"]
