import tensorflow as tf

from surveillance_tf.losses.mil_losses import compute_losses


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
