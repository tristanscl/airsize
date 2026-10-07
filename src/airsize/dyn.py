import numpy as np


def compute_n(V, g0, RC):
    return np.sqrt(1 + (V**2 / (g0 * RC)) ** 2)


def compute_RC(V, g0, n):
    return V**2 / (g0 * np.sqrt(n**2 - 1))


def compute_N(t, V, n, RC):
    return t * V / (2 * n * RC)
