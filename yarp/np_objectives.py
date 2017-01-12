import numpy as np


def mse(a, b):
    return np.square(a - b).sum(axis=(1, 2, 3))
