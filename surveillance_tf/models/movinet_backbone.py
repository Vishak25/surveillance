"""MoViNet backbone from TF-Hub."""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf
import tensorflow_hub as hub

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

_MOVINET_URL = "https://tfhub.dev/tensorflow/movinet/a0/feature_vector/3"


def build_segment_encoder(
    input_shape: Tuple[int | None, int, int, int] = (None, 224, 224, 3),
    trainable: bool = True,
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    """Return a Keras model mapping video segments to feature vectors."""
    inputs = tf.keras.Input(shape=input_shape, name="segments")
    hub_layer = hub.KerasLayer(_MOVINET_URL, trainable=trainable, name="movinet")
    features = hub_layer(inputs)
    if dropout_rate > 0:
        features = tf.keras.layers.Dropout(dropout_rate)(features)
    model = tf.keras.Model(inputs=inputs, outputs=features, name="movinet_encoder")
    LOGGER.info("Constructed MoViNet encoder (trainable=%s)", trainable)
    return model


__all__ = ["build_segment_encoder"]
