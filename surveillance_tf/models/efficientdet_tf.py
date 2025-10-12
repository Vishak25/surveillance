"""EfficientDet-Lite wrapper for overlay detections."""
from __future__ import annotations

from typing import Dict

import tensorflow as tf
import tensorflow_hub as hub

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

_DET_URLS = {
    "lite0": "https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1",
    "lite2": "https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1",
}


def load_detector(variant: str = "lite0"):
    """Load EfficientDet-Lite model from TF-Hub."""
    if variant not in _DET_URLS:
        raise ValueError(f"Unsupported EfficientDet variant: {variant}")
    LOGGER.info("Loading EfficientDet-%s from TF-Hub", variant)
    model = hub.load(_DET_URLS[variant])
    return model


def run_detector(model, image_tensor: tf.Tensor, threshold: float = 0.3) -> Dict[str, tf.Tensor]:
    """Run detector on a single image tensor and filter by score."""
    outputs = model(image_tensor)
    scores = outputs["detection_scores"]
    mask = scores >= threshold
    filtered = {k: tf.boolean_mask(v, mask) for k, v in outputs.items() if v.shape.rank > 0}
    LOGGER.debug(
        "Detector returned %d boxes above threshold %.2f",
        tf.shape(filtered["detection_scores"])[0],
        threshold,
    )
    return filtered


__all__ = ["load_detector", "run_detector"]
