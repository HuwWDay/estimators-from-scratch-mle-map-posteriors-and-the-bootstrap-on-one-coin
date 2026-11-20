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

# Step 6 - map_bernoulli
def map_bernoulli(N1, N0, a, b):
    # TODO: Return the mode of the posterior Beta(N1 + a, N0 + b) as a float.
    #       Raise ValueError if N1 + a < 1, N0 + b < 1 or N1 + N0 + a + b <= 2.
    if N1 + a < 1 or N0 + b < 1 or N1 + N0 + a + b <= 2:
        raise ValueError
    return (N1+a-1)/(N1+N0+a+b-2)

# Step 7 - log_loss
import numpy as np


def log_loss(p_heads, y):
    # TODO: Average Bernoulli negative log likelihood (nats per flip) of the flips y under p_heads.
    #       Reuse coin_counts and bernoulli_nll; return a Python float.
    N1, N0 = coin_counts(y)
    return bernoulli_nll(p_heads, N1, N0)/(N0+N1)

# Step 8 - black_swan_risk
import math
import numpy as np


def black_swan_risk(theta, n, n_future, n_rep, rng):
    mle_inf_count = 0
    mle_finite_losses = []
    add_one_losses = []
    oracle_losses = []

    for _ in range(n_rep):
        y = (rng.random(n) < theta).astype(int)
        future = (rng.random(n_future) < theta).astype(int)

        N1, N0 = coin_counts(y)

        p_mle = N1 / n
        p_add_one = map_bernoulli(N1, N0, 2, 2)

        loss_mle = log_loss(p_mle, future)
        if math.isinf(loss_mle):
            mle_inf_count += 1
        else:
            mle_finite_losses.append(loss_mle)

        add_one_losses.append(log_loss(p_add_one, future))
        oracle_losses.append(log_loss(theta, future))

    mle_loss_when_finite = (
        float(np.mean(mle_finite_losses)) if mle_finite_losses else float("nan")
    )

    return {
        "mle_infinite": float(mle_inf_count / n_rep),
        "mle_loss_when_finite": mle_loss_when_finite,
        "add_one_loss": float(np.mean(add_one_losses)),
        "oracle_loss": float(np.mean(oracle_losses)),
    }

# Step 9 - beta_logpdf
import math
import numpy as np


def log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_logpdf(theta, a, b):
    is_scalar = np.isscalar(theta)
    t = np.asarray(theta, dtype=float)
    in_bounds = (t >= 0.0) & (t <= 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(a == 1, 0.0, (a - 1.0) * np.log(t))
        term2 = np.where(b == 1, 0.0, (b - 1.0) * np.log(1.0 - t))
        val = term1 + term2 - log_beta(a, b)

    res = np.where(in_bounds, val, -np.inf)
    return float(res) if is_scalar else res

# Step 10 - beta_posterior
def beta_posterior(N1, N0, a, b):
    a_ = a + N1
    b_ = b + N0
    N = N1 + N0

    try:
        mode = float(map_bernoulli(N1, N0, a, b))
    except ValueError:
        mode = None

    return {
        "a": float(a_),
        "b": float(b_),
        "mean": float(a_ / (a_ + b_)),
        "mode": mode,
        "var": float((a_ * b_) / ((a_ + b_) ** 2 * (a_ + b_ + 1))),
        "prior_weight": float((a + b) / (a + b + N)),
    }

# Step 11 - log_evidence
import math


def log_evidence(N1, N0, a, b, binomial=False):
    # TODO: log B(a + N1, b + N0) - log B(a, b), plus log C(N, N1) when binomial=True.
    out = log_beta(a+N1, b+N0) - log_beta(a, b)
    if binomial:
        return out + math.log(math.comb(N0+N1, N1))
    else:
        return out

# Step 12 - sequential_predictive
import numpy as np


def sequential_predictive(y, a, b):
    coin_counts(y)
    y = np.asarray(y)

    heads_before = np.concatenate(([0], np.cumsum(y[:-1])))
    t = np.arange(y.size)
    return (heads_before + a) / (t + a + b)

