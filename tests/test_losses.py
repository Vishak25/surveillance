import tensorflow as tf

from surveillance_tf.losses.mil_losses import (
    compute_losses,
    compute_one_class_mil_losses,
)


def test_ranking_loss_zero_when_margin_satisfied():
    pos = tf.ragged.constant([[0.9, 0.8]])
    neg = tf.ragged.constant([[0.1, 0.2]])
    losses = compute_losses(pos, neg, margin=0.5)
    assert tf.math.is_finite(losses["ranking"])
    assert losses["ranking"].numpy() == 0.0


def test_sparsity_loss_equals_mean():
    pos = tf.ragged.constant([[0.25, 0.75]])
    neg = tf.ragged.constant([[0.1, 0.2]])
    losses = compute_losses(pos, neg)
    assert abs(losses["sparsity"].numpy() - 0.5) < 1e-6


def test_smoothness_loss_zero_for_constant_segments():
    pos = tf.ragged.constant([[0.4, 0.4, 0.4]])
    neg = tf.ragged.constant([[0.1, 0.1, 0.1]])
    losses = compute_losses(pos, neg)
    assert losses["smoothness"].numpy() == 0.0


def test_one_class_constant_scores_smoothness_zero_and_sparsity_mean():
    bag_scores = tf.ragged.constant([[0.5, 0.5, 0.5]])
    losses = compute_one_class_mil_losses(bag_scores, k=2)
    assert losses["smoothness"].numpy() == 0.0
    assert abs(losses["sparsity"].numpy() - 0.5) < 1e-6


def test_one_class_ranking_with_top_bottom_k():
    scores = tf.ragged.constant([[0.1, 0.5, 0.9]])
    losses = compute_one_class_mil_losses(scores, k=1, margin=1.0, lambda_sparse=0.0, lambda_smooth=0.0)
    expected = max(0.0, 1.0 - 0.9 + 0.1)
    assert abs(losses["ranking"].numpy() - expected) < 1e-6


def test_one_class_total_is_scalar_and_finite():
    scores = tf.ragged.constant([[0.2, 0.4, 0.6], [0.9]])
    losses = compute_one_class_mil_losses(scores)
    total = losses["total"]
    assert total.shape == ()
    assert bool(tf.math.is_finite(total))
