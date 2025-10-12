"""SORT tracker with Hungarian matching."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class Track:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    hits: int = 0
    time_since_update: int = 0
    history: List[np.ndarray] = field(default_factory=list)


class SortTracker:
    def __init__(self, max_age: int = 10, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self._next_id = 1
        self._dt = 1.0
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = self._dt
        self._H = np.eye(4, 8)
        self._Q = np.eye(8) * 1e-2
        self._R = np.eye(4) * 1e-1

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        return np.array([x, y, w, h], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        x, y, w, h = z
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def _predict(self, track: Track) -> None:
        track.state = self._F @ track.state
        track.covariance = self._F @ track.covariance @ self._F.T + self._Q
        track.time_since_update += 1
        track.history.append(self._z_to_bbox(track.state[:4]))
        if len(track.history) > self.max_age:
            track.history.pop(0)

    def _update_track(self, track: Track, measurement: np.ndarray) -> None:
        z = self._bbox_to_z(measurement)
        y = z - self._H @ track.state
        S = self._H @ track.covariance @ self._H.T + self._R
        K = track.covariance @ self._H.T @ np.linalg.pinv(S)
        track.state = track.state + K @ y
        I = np.eye(self._H.shape[1])
        track.covariance = (I - K @ self._H) @ track.covariance
        track.hits += 1
        track.time_since_update = 0
        track.history.append(measurement)

    def _init_track(self, bbox: np.ndarray) -> Track:
        state = np.zeros(8, dtype=np.float32)
        state[:4] = self._bbox_to_z(bbox)
        covariance = np.eye(8, dtype=np.float32)
        track = Track(track_id=self._next_id, state=state, covariance=covariance)
        self._next_id += 1
        return track

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Update tracker with detections array [[x1,y1,x2,y2], ...]."""
        detections = np.asarray(detections, dtype=np.float32)
        for track in self.tracks:
            self._predict(track)

        cost_matrix = None
        if detections.size == 0 or len(self.tracks) == 0:
            matched_indices = np.empty((0, 2), dtype=int)
        else:
            predicted_boxes = np.array([self._z_to_bbox(t.state[:4]) for t in self.tracks])
            cost_matrix = np.zeros((len(predicted_boxes), len(detections)), dtype=np.float32)
            for i, pbox in enumerate(predicted_boxes):
                for j, det in enumerate(detections):
                    cost_matrix[i, j] = 1.0 - self._iou(pbox, det)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_indices = np.array(list(zip(row_ind, col_ind)), dtype=int)

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))
        for t_idx, d_idx in matched_indices:
            if cost_matrix is not None and cost_matrix[t_idx, d_idx] > 1.0 - self.iou_threshold:
                continue
            self._update_track(self.tracks[t_idx], detections[d_idx])
            unmatched_tracks.discard(t_idx)
            unmatched_detections.discard(d_idx)

        for idx in sorted(unmatched_detections):
            new_track = self._init_track(detections[idx])
            self.tracks.append(new_track)

        self.tracks = [t for i, t in enumerate(self.tracks) if i not in unmatched_tracks or t.time_since_update <= self.max_age]

        outputs = []
        for track in self.tracks:
            if track.hits >= self.min_hits or track.time_since_update == 0:
                bbox = self._z_to_bbox(track.state[:4])
                outputs.append(np.concatenate([bbox, [track.track_id]], axis=0))
        return np.array(outputs, dtype=np.float32)


__all__ = ["SortTracker"]
