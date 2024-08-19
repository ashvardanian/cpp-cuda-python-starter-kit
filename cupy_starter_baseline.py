import numpy as np
import warnings
from numba import njit


def safe_njit(f):
    try:
        return njit(f)
    except Exception as e:
        warnings.warn(f"Numba JIT compilation failed for function {f.__name__}: {e}")
        return f


@safe_njit
def matmul_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Equivalent to np.matmul(a, b)"""
    m, n = a.shape
    n, k = b.shape
    c = np.zeros((m, k), dtype=a.dtype)
    for i in range(m):
        for j in range(k):
            for l in range(n):
                c[i, j] += a[i, l] * b[l, j]
    return c


@safe_njit
def reduce_numba(a: np.ndarray) -> float:
    """Equivalent to np.sum(a)"""
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i]
    return s


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a, b)


def reduce(a: np.ndarray) -> float:
    return np.sum(a)
