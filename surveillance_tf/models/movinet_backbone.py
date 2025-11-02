"""MoViNet backbone resolved via KaggleHub cache."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from keras.layers import TFSMLayer
import kagglehub

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

_MOVINET_MODEL_HANDLE = "google/movinet/tensorFlow2/a0-base-kinetics-600-classification"
_MOVINET_ENV_OVERRIDE = "MOVINET_MODEL_DIR"


@lru_cache(maxsize=1)
def _resolve_movinet_path() -> Path:
    """Return local directory containing the MoViNet SavedModel."""
    override = os.getenv(_MOVINET_ENV_OVERRIDE)
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(
                f"MOVINET model override '{candidate}' does not exist. "
                "Set MOVINET_MODEL_DIR to the directory containing saved_model.pb."
            )
        LOGGER.info("Using MoViNet model from MOVINET_MODEL_DIR=%s", candidate)
        return candidate

    LOGGER.info("Downloading MoViNet model via KaggleHub handle '%s'", _MOVINET_MODEL_HANDLE)
    path = Path(kagglehub.model_download(_MOVINET_MODEL_HANDLE)).resolve()
    LOGGER.info("MoViNet model cached at %s", path)
    return path


def _build_movinet_layer(trainable: bool) -> TFSMLayer:
    path = _resolve_movinet_path()
    # SavedModel exposes `classifier_head` as its default output. Treat it as feature vector input to MIL head.
    return TFSMLayer(str(path), call_endpoint="serving_default", trainable=trainable, name="movinet_backbone")


def build_segment_encoder(
    input_shape: Tuple[int | None, int, int, int] = (None, 224, 224, 3),
    trainable: bool = True,
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    """Return a Keras model mapping video segments to feature vectors."""
    inputs = tf.keras.Input(shape=input_shape, name="segments")
    movinet_layer = _build_movinet_layer(trainable)
    outputs = movinet_layer(inputs)
    if isinstance(outputs, dict):
        # Default signature returns a dict with key 'classifier_head'.
        features = outputs.get("classifier_head")
        if features is None:
            raise KeyError("MoViNet SavedModel did not provide 'classifier_head' in its outputs.")
    else:
        features = outputs
    if dropout_rate > 0:
        features = tf.keras.layers.Dropout(dropout_rate)(features)
    model = tf.keras.Model(inputs=inputs, outputs=features, name="movinet_encoder")
    LOGGER.info("Constructed MoViNet encoder (trainable=%s)", trainable)
    return model


__all__ = ["build_segment_encoder"]
