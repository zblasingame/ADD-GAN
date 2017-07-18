"""A module for creating Neural Nets

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import tensorflow as tf
from functools import reduce


def build_residual_block(X, act, num_inputs, reg_param=0.01):
    """Builds a resiudal learning block.

    Args:
        X (tf.Tensor): Input, rank one Tensor.
        act (function): Activation function.
        num_inputs (int): Number of inputs.
        reg_param (float = 0.01): L2 Regularization parameter.

    Returns:
        tf.Tensor: Output of residual net.
    """

    # initializer = tf.random_normal_initializer(mean=0, stddev=0.1)
    initializer = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(reg_param)

    with tf.variable_scope('residual_block'):

        w1 = tf.get_variable(
            'w1',
            [num_inputs, num_inputs],
            initializer=initializer,
            regularizer=regularizer
        )
        b1 = tf.get_variable(
            'b1',
            [num_inputs],
            initializer=initializer,
            regularizer=regularizer
        )

        w2 = tf.get_variable(
            'w2',
            [num_inputs, num_inputs],
            initializer=initializer,
            regularizer=regularizer
        )
        b2 = tf.get_variable(
            'b2',
            [num_inputs],
            initializer=initializer,
            regularizer=regularizer
        )

    f = act(tf.add(tf.matmul(X, w1), b1))

    return act(tf.add(tf.add(tf.matmul(f, w2), b2), X))


def build_fc_layer(X, act, input_dim, output_dim, reg_param=0.01):
    """Builds a fully connected layer

    Args:
        X (tf.Tensor): Input, rank one Tensor.
        act (function): Activation function.
        input_dim (int): Size of input dimension.
        output_dim (int): Size of output dimension.
        reg_param (float = 0.01): L2 Regularization parameter.


    Returns:
        tf.Tensor: Output of fc layer.
    """

    # initializer = tf.random_normal_initializer(mean=0, stddev=0.1)
    initializer = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(reg_param)

    with tf.variable_scope('fc_layer'):
        w1 = tf.get_variable(
            'w1',
            [input_dim, output_dim],
            initializer=initializer,
            regularizer=regularizer
        )
        b1 = tf.get_variable(
            'b1',
            [output_dim],
            initializer=initializer,
            regularizer=regularizer
        )

    return act(tf.add(tf.matmul(X, w1), b1))


def lrelu_gen(alpha):
    """Returns a Leaky ReLU

    Args:
        alpha (float): Leaking parameter.

    Returns:
        function: LReLU
    """

    def lrelu(x):
        """Leaky ReLU

        Args:
            x (tf.Tensor): Input Tensor.
            alpha (float): Leaking parameter.

        Returns:
            tf.Tensor: Output of LReLU
        """

        return tf.maximum(alpha * x, x)

    return lrelu


class NeuralNet:
    """A basic neural net implementation.

    Args:
        sizes (list of [int]): List describing the number of
            units in each layer.
        activations (list of [function]): List of TensorFlow activation
            functions, must have one less element than the
            number of elements in the parameter sizes.
    """

    def __init__(self, sizes, activations):
        """Initializes NeuralNet class."""
        assert len(sizes) == len(activations) + 1, (
            'sizes and activations have a missmatched number of elements'
        )

        def create_weights(shape, name):
            return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)

        self.network = [{
            'weights': create_weights([sizes[i], sizes[i+1]], 'w' + str(i)),
            'biases': create_weights([sizes[i+1]], 'b' + str(i)),
            'activation': activations[i]
        } for i in range(len(sizes) - 1)]

    def create_network(self, X, keep_prob=tf.constant(1.0)):
        """Method to construct the network.

        Args:
            X (tf.Tensor): Placeholder Tensor with dimenions of the
                training Tensor.
            keep_prob (tf.Tensor = tf.constant(1.0)): Placeholder Tensor
                of rank one of the probability for
                the dropout technique.

        Returns:
            tf.Tensor: A tensor to be evaulated containing the predicted
                output of the neural net.
        """

        def compose_func(a, x, w, b):
            return a(tf.matmul(x, w) + b)

        prev_value = X
        for i, entry in enumerate(self.network):
            prev_value = compose_func(entry['activation'],
                                      prev_value,
                                      entry['weights'],
                                      entry['biases'])

            if i != len(self.network) - 1:
                prev_value = tf.nn.dropout(prev_value, keep_prob)

        return prev_value

    def reset_weights(self):
        """Returns a TensorFlow operation to resets TensorFlow weights
        so the model can be used again.

        Returns:
            list [tf.Operation]: List of operations to reassign weights.
        """

        weights = [entry['weights'] for entry in self.network]
        weights.extend([entry['biases'] for entry in self.network])

        return [weight.assign(tf.random_normal(weight.get_shape(), stddev=0.1))
                for weight in weights]

    def get_l2_loss(self):
        """Method to return the L2 loss for L2 regularization techniques.

        Returns:
            tf.Tensor: A tensor to be evaulated containing the
                L2 loss of the network.
        """

        weights = [entry['weights'] for entry in self.network]
        weights.extend([entry['biases'] for entry in self.network])

        return reduce(lambda a, b: a + tf.nn.l2_loss(b), weights, 0)
