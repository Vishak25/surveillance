"""Inference fusion utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def predict_with_ci(model: tf.keras.Model, segments: tf.Tensor, passes: int = 10) -> Tuple[float, float]:
    """Monte Carlo dropout predictions returning mean and std of max scores."""
    scores = []
    for _ in range(passes):
        preds = tf.squeeze(model(segments, training=True), axis=-1)
        max_score = float(tf.reduce_max(preds).numpy()) if tf.size(preds) > 0 else 0.0
        scores.append(max_score)
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    LOGGER.debug("MC dropout mean %.3f std %.3f", mean, std)
    return mean, std


def fuse_scores(anomaly_score: float) -> float:
    """Placeholder fusion returning anomaly score (single-stream)."""
    return anomaly_score


__all__ = ["predict_with_ci", "fuse_scores"]
