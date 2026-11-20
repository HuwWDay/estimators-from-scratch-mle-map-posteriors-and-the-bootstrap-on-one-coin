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

