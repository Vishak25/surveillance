"""Deterministic seeding utilities."""
from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf

from .logging import get_logger

LOGGER = get_logger(__name__)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow PRNGs.

    TensorFlow determinism is best-effort: sets CUDA deterministic ops when available.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:  # TensorFlow deterministic controls (GPU only)
        from tensorflow.keras import backend as K  # pylint: disable=import-outside-toplevel

        K.clear_session()
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    except Exception as exc:  # pragma: no cover - optional dependency paths
        LOGGER.warning("Could not fully enforce TF determinism: %s", exc)


__all__ = ["set_global_seed"]
