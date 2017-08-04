"""Module for loading and parsing training data.

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import csv
import numpy as np


def load_data(filename):
    """Returns the features of a dataset.

    Args:
        filename (str): File location (csv formatted).

    Returns:
        tuple of np.ndarray: Tuple consisiting of the features,
            X, and the labels, Y.
    """

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = np.array([row for row in reader
                         if '#' not in row[0]]).astype(np.float32)

    X = data[:, 1:]
    Y = data[:, 0]

    Y = np.clip(Y, 0, 1)

    return X, Y


def batcher(data, batch_size=100):
    """Creates a generator to yield batches of batch_size.
    When batch is too large to fit remaining data the batch
    is clipped.

    Args:
        data (List of np.ndarray): List of data elements to be batched.
            The first dimension must be the batch size and the same
            for all data elements.
        batch_size (int = 100): Size of the mini batches.
    Yields:
        The next mini_batch in the dataset.
    """

    batch_start = 0
    batch_end   = batch_size

    while batch_end < data[0].shape[0]:
        yield [el[batch_start:batch_end] for el in data]

        batch_start = batch_end
        batch_end   += batch_size

    yield [el[batch_start:] for el in data]


def random_batcher(data, batch_size=100):
    """Creates a generator to yield random mini-batches of batch_size.
    When batch is too large to fit remaining data the batch
    is clipped. Will continously cycle through data.

    Args:
        data (List of np.ndarray): List of data elements to be batched.
            The first dimension must be the batch size and the same
            for all data elements.
        batch_size (int = 100): Size of the mini batches.
    Yields:
        The next mini_batch in the dataset.
    """

    while True:
        for el in data:
            np.random.shuffle(el)

        batch_start = 0
        batch_end   = batch_size

        while batch_end < data[0].shape[0]:
            yield [el[batch_start:batch_end] for el in data]

            batch_start = batch_end
            batch_end   += batch_size

        yield [el[batch_start:] for el in data]


def rescaling(data, _min, _max, start=0.0, end=1.0, axis=0):
    """Rescale features of a dataset

    args:
        data (np.array): feature matrix.
        _min (np.array): list of minimum values per feature.
        _max (np.array): list of maximum values per feature.
        start (float = 0.): lowest value for norm.
        end (float = 1.): highest value for norm.
        axis (int = 0): axis to normalize across

    returns:
        (np.array): normalized features, the same shape as data
    """

    new_data = (data - _min) / (_max - _min)

    # check if feature is constant, will be nan in new_data
    np.place(new_data, np.isnan(new_data), 1)

    new_data = (end - start) * new_data + start

    return new_data


def vector_norm(data, start=0.0, end=1.0):
    """Scaling feature vectors

    args:
        data (np.array): feature matrix.
        _min (np.array): list of minimum values per feature.
        _max (np.array): list of maximum values per feature.
        start (float = 0.): lowest value for norm.
        end (float = 1.): highest value for norm.
        axis (int = 0): axis to normalize across

    returns:
        (np.array): normalized features, the same shape as data
    """

    new_data = data / np.sqrt(np.sum(data * data, axis=1))[:, None]
    new_data = (end - start) * new_data + start

    return new_data
