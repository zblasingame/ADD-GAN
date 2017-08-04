"""Module for tensorboard ops.
From TensorFlow Tutorial see
https://www.tensorflow.org/get_started/summaries_and_tensorboard

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import tensorflow as tf


def variable_summaries(var):
    """Attatch summaries of a variable to a Tensor for TensorBoard.

    Args:
        var (tf.Tensor): Tensor variable.
    """

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
