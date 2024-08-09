import numpy as np
from numba import njit


@njit
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m, n = a.shape
    n, k = b.shape
    c = np.zeros((m, k), dtype=a.dtype)
    for i in range(m):
        for j in range(k):
            for l in range(n):
                c[i, j] += a[i, l] * b[l, j]
    return c


@njit
def reduce(a: np.ndarray) -> float:
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i]
    return s
