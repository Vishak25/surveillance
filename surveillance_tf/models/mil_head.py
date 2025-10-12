"""MIL scoring head."""
from __future__ import annotations

import tensorflow as tf

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class MILScoringHead(tf.keras.Model):
    """Multi-layer perceptron that outputs anomaly score in [0, 1]."""

    def __init__(self, units: list[int] | None = None, name: str = "mil_head") -> None:
        super().__init__(name=name)
        units = units or [256, 128]
        self.denses = []
        self.dropouts = []
        for idx, unit in enumerate(units):
            self.denses.append(tf.keras.layers.Dense(unit, activation="relu", name=f"dense_{idx}"))
            self.dropouts.append(tf.keras.layers.Dropout(0.2, name=f"dropout_{idx}"))
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="score")
        LOGGER.info("MILScoringHead initialized with units=%s", units)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:  # type: ignore[override]
        x = inputs
        for dense, drop in zip(self.denses, self.dropouts):
            x = dense(x)
            x = drop(x, training=training)
        score = self.output_layer(x)
        return score


__all__ = ["MILScoringHead"]
