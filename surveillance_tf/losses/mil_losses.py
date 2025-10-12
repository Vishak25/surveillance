"""Multiple-instance learning losses for anomaly detection."""
from __future__ import annotations

from typing import Dict

import tensorflow as tf

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def _ensure_ragged(t: tf.RaggedTensor | tf.Tensor) -> tf.RaggedTensor:
    return t if isinstance(t, tf.RaggedTensor) else tf.RaggedTensor.from_tensor(t)


def compute_losses(
    pos_scores: tf.RaggedTensor | tf.Tensor,
    neg_scores: tf.RaggedTensor | tf.Tensor,
    margin: float = 1.0,
    lambda_sparse: float = 8e-5,
    lambda_smooth: float = 0.1,
) -> Dict[str, tf.Tensor]:
    r"""Compute ranking, sparsity, and smoothness losses.

    Ranking loss:
    .. math:: L_{rank} = \mathbb{E}[\max(0, m - \max_t s_p^t + \max_t s_n^t)]

    Sparsity loss:
    .. math:: L_{sparse} = \mathbb{E}_p \left[ \frac{1}{|p|} \sum_t s_p^t \right]

    Smoothness loss:
    .. math:: L_{smooth} = \mathbb{E}\left[ \frac{1}{T-1} \sum_t (s^t - s^{t+1})^2 \right]
    """
    pos_rt = _ensure_ragged(pos_scores)
    neg_rt = _ensure_ragged(neg_scores)

    if pos_rt.nrows() == 0 or neg_rt.nrows() == 0:
        zero = tf.constant(0.0, dtype=tf.float32)
        return {"ranking": zero, "sparsity": zero, "smoothness": zero, "total": zero}

    pos_max = pos_rt.reduce_max(axis=1)  # [N_pos]
    neg_max = neg_rt.reduce_max(axis=1)  # [N_neg]
    diff = margin - tf.expand_dims(pos_max, axis=1) + tf.expand_dims(neg_max, axis=0)
    ranking = tf.reduce_mean(tf.nn.relu(diff))

    pos_means = pos_rt.reduce_mean(axis=1)
    sparsity = tf.reduce_mean(pos_means)

    diffs = pos_rt[:, 1:] - pos_rt[:, :-1]
    neg_diffs = neg_rt[:, 1:] - neg_rt[:, :-1]
    all_diffs = tf.concat([diffs.flat_values, neg_diffs.flat_values], axis=0)
    if tf.size(all_diffs) == 0:
        smoothness = tf.constant(0.0, dtype=tf.float32)
    else:
        smoothness = tf.reduce_mean(tf.square(all_diffs))

    total = ranking + lambda_sparse * sparsity + lambda_smooth * smoothness
    LOGGER.debug(
        "Losses -> ranking: %.5f, sparsity: %.5f, smoothness: %.5f, total: %.5f",
        ranking,
        sparsity,
        smoothness,
        total,
    )
    return {
        "ranking": ranking,
        "sparsity": sparsity,
        "smoothness": smoothness,
        "total": total,
    }


__all__ = ["compute_losses"]
