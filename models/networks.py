"""A module for building neural nets

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import tensorflow as tf
import models.nnblocks as nn


def gen_res_net(sizes):
    """Builds a resiudal learning block.

    Args:
        sizes (list[int]): Size of each layer.

    Returns:
        (function): Function to build network.
    """

    def build_net(X, keep_prob=tf.constant(1.0)):
        lrelu   = nn.lrelu_gen(0.1)

        def block(x, in_dim, out_dim, i):
            with tf.variable_scope('block_{}'.format(i)):
                z = x
                for j in range(2):
                    with tf.variable_scope('res_block_{}'.format(j)):
                        z = nn.build_residual_block(z, lrelu, in_dim)
                        z = tf.nn.dropout(z, keep_prob)

                z = nn.build_fc_layer(z, lrelu, in_dim, out_dim)

                return tf.nn.dropout(z, keep_prob)
        z = X

        for i in range(1, len(sizes)):
            z = block(z, sizes[i-1], sizes[i], i-1)

        return z

    return build_net
