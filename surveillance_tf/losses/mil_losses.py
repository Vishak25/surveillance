"""Multiple-instance learning losses for anomaly detection."""
from __future__ import annotations

from typing import Dict

import tensorflow as tf

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def _ensure_ragged(t: tf.RaggedTensor | tf.Tensor) -> tf.RaggedTensor:
    return t if isinstance(t, tf.RaggedTensor) else tf.RaggedTensor.from_tensor(t)


def _ragged_to_dense_with_lengths(t: tf.RaggedTensor) -> tuple[tf.Tensor, tf.Tensor]:
    dense = t.to_tensor()
    lengths = tf.cast(t.row_lengths(), tf.int32)
    return dense, lengths


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


def compute_one_class_mil_losses(
    bag_scores: tf.RaggedTensor | tf.Tensor,
    k: int = 3,
    margin: float = 1.0,
    lambda_sparse: float = 8e-5,
    lambda_smooth: float = 0.1,
) -> dict:
    r"""One-Class MIL loss composed of ranking, sparsity, and smoothness terms.

    For bag :math:`i` with segment scores :math:`s_{i,t} \in [0, 1]`:

    .. math::
        \begin{aligned}
        \text{top}_i &= \frac{1}{k_i} \sum_{t \in \text{TopK}(s_i)} s_{i,t}, \\
        \text{bot}_i &= \frac{1}{k_i} \sum_{t \in \text{BottomK}(s_i)} s_{i,t}, \\
        L^{(i)}_{\text{rank}} &= \max\left(0, m - \text{top}_i + \text{bot}_i\right), \\
        L^{(i)}_{\text{sparse}} &= \frac{1}{|s_i|} \sum_t s_{i,t}, \\
        L^{(i)}_{\text{smooth}} &= \frac{1}{|s_i|-1} \sum_t (s_{i,t} - s_{i,t+1})^2,
        \end{aligned}
    where :math:`k_i = \min(k, |s_i|)`. The total loss averages each term across the batch:

    .. math:: L = \bar{L}_{\text{rank}} + \lambda_{\text{sparse}}\bar{L}_{\text{sparse}} + \lambda_{\text{smooth}}\bar{L}_{\text{smooth}}.
    """
    scores_rt = _ensure_ragged(bag_scores)
    if scores_rt.nrows() == 0:
        zero = tf.constant(0.0, dtype=tf.float32)
        return {"ranking": zero, "sparsity": zero, "smoothness": zero, "total": zero}
    dense, lengths = _ragged_to_dense_with_lengths(scores_rt)

    def per_bag(values: tf.Tensor, length: tf.Tensor) -> tf.Tensor:
        length = tf.maximum(length, 0)
        k_eff = tf.maximum(1, tf.minimum(k, length))

        values = tf.cast(values[:length], tf.float32)

        def _safe_mean(x: tf.Tensor) -> tf.Tensor:
            return tf.cond(
                tf.size(x) > 0,
                lambda: tf.reduce_mean(x),
                lambda: tf.constant(0.0, dtype=tf.float32),
            )

        sorted_desc = tf.sort(values, direction="DESCENDING")
        sorted_asc = tf.sort(values, direction="ASCENDING")

        top = _safe_mean(sorted_desc[:k_eff])
        bot = _safe_mean(sorted_asc[:k_eff])
        sparsity_val = _safe_mean(values)

        smooth = tf.cond(
            length > 1,
            lambda: tf.reduce_mean(tf.square(values[1:] - values[:-1])),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )
        return tf.stack([top, bot, sparsity_val, smooth], axis=0)

    per_bag_stats = tf.map_fn(
        lambda args: per_bag(args[0], args[1]),
        (dense, lengths),
        dtype=tf.float32,
    )
    top_vals = per_bag_stats[:, 0]
    bot_vals = per_bag_stats[:, 1]
    sparsity_vals = per_bag_stats[:, 2]
    smooth_vals = per_bag_stats[:, 3]

    ranking = tf.reduce_mean(tf.nn.relu(margin - top_vals + bot_vals))
    sparsity = tf.reduce_mean(sparsity_vals)
    smoothness = tf.reduce_mean(smooth_vals)

    total = ranking + lambda_sparse * sparsity + lambda_smooth * smoothness
    LOGGER.debug(
        "One-Class MIL losses -> ranking: %.5f, sparsity: %.5f, smoothness: %.5f, total: %.5f",
        ranking,
        sparsity,
        smoothness,
        total,
    )
    return {"ranking": ranking, "sparsity": sparsity, "smoothness": smoothness, "total": total}


__all__ = ["compute_losses", "compute_one_class_mil_losses"]
