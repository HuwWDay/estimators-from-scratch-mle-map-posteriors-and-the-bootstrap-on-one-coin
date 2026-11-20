"""
Estimators from Scratch: MLE, MAP, Posteriors and the Bootstrap on One Coin

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - coin_counts
import numpy as np


def coin_counts(y):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("Input must be a 1-D array.")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Input must contain only 0s and 1s.")

    N1 = int(np.sum(y))
    N0 = int(y.size - N1)
    return N1, N0

# Step 2 - bernoulli_nll
import numpy as np


def bernoulli_nll(theta, N1, N0):
    is_scalar = np.isscalar(theta)
    theta = np.asarray(theta, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(N1 == 0, 0.0, N1 * np.log(theta))
        term2 = np.where(N0 == 0, 0.0, N0 * np.log(1.0 - theta))

    nll = -(term1 + term2)
    return float(nll) if is_scalar else nll

# Step 3 - golden_section_min
import math


def golden_section_min(f, lo, hi, tol=1e-10):
    a, b = float(lo), float(hi)
    r = (math.sqrt(5) - 1) / 2

    diff = r * (b - a)
    c = b - diff
    d = a + diff
    fc = f(c)
    fd = f(d)

    while (b - a) >= tol:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - r * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + r * (b - a)
            fd = f(d)

    return float((a + b) / 2)

# Step 4 - categorical_mle
import numpy as np


def categorical_mle(y, K):
    y = np.asarray(y)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a non-empty 1-D array.")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y must contain integer outcomes.")
    if np.any((y < 0) | (y >= K)):
        raise ValueError(f"All outcomes in y must be in 0..{K - 1}.")

    counts = np.bincount(y, minlength=K)
    probs = counts / y.size
    return counts, probs

# Step 5 - gaussian_mle
import numpy as np


def gaussian_mle(x):
    # TODO: Raise ValueError for fewer than 2 readings. From the sufficient statistics
    #       mean(x) and mean(x**2), return (mu, var_mle (divide by N), var_unbiased (divide by N-1)).
    x = np.asarray(x)
    N = len(x)
    if N < 2:
        raise ValueError
    mu = np.mean(x)
    s2 = np.sum((x-mu)**2) 
    return float(mu), float(s2/N), float(s2/(N-1))

