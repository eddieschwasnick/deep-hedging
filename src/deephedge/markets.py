"""Market simulators — the 'world' your hedger lives in.

Everything is vectorized: one call returns thousands of price paths at once,
shaped (n_paths, n_steps+1). That shape is the spine of the whole project.

You only ever need TWO functions here to start:
    simulate_gbm(...)     -> Black-Scholes stock paths
    bs_price_delta(...)   -> the textbook price and hedge ratio to compare against
Heston comes later (Week 2); it's here so you can peek.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Black-Scholes world
# ---------------------------------------------------------------------------
def simulate_gbm(s0, mu, sigma, T, n_steps, n_paths, seed=0):
    """Geometric Brownian motion: dS = mu S dt + sigma S dW.

    Returns S of shape (n_paths, n_steps+1). Column 0 is s0 for every path.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    # log-increments are normal; exact (no discretization error in the SDE).
    z = rng.standard_normal((n_paths, n_steps))
    log_incr = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_incr, axis=1)], axis=1
    )
    return s0 * np.exp(log_paths)


def _norm_cdf(x):
    from math import erf, sqrt
    # vectorized standard normal CDF via erf
    return 0.5 * (1.0 + np.vectorize(lambda v: erf(v / np.sqrt(2)))(x))


def bs_price_delta(S, K, T_remaining, sigma, r=0.0):
    """Black-Scholes call price and delta.

    S, T_remaining can be arrays (same shape). Returns (price, delta).
    delta = dPrice/dS = the number of shares the textbook says to hold.
    """
    S = np.asarray(S, dtype=float)
    T_remaining = np.asarray(T_remaining, dtype=float)
    eps = 1e-12
    Tr = np.maximum(T_remaining, eps)
    d1 = (np.log(np.maximum(S, eps) / K) + (r + 0.5 * sigma**2) * Tr) / (sigma * np.sqrt(Tr))
    d2 = d1 - sigma * np.sqrt(Tr)
    price = S * _norm_cdf(d1) - K * np.exp(-r * Tr) * _norm_cdf(d2)
    delta = _norm_cdf(d1)
    # at expiry, delta is a step function; price is the payoff
    at_expiry = T_remaining <= eps
    price = np.where(at_expiry, np.maximum(S - K, 0.0), price)
    delta = np.where(at_expiry, (S > K).astype(float), delta)
    return price, delta


# ---------------------------------------------------------------------------
# Heston world (stochastic volatility) — for Week 2
# ---------------------------------------------------------------------------
def simulate_heston(s0, v0, alpha, b, sigma_v, rho, T, n_steps, n_paths, seed=0):
    """Heston model with full-truncation Euler for the variance.

    dS = sqrt(V) S dB
    dV = alpha (b - V) dt + sigma_v sqrt(V) dW,   corr(dB, dW) = rho

    Returns (S, V), each (n_paths, n_steps+1). Variance is clipped at 0 so it
    never goes negative (the classic Heston-simulation gotcha).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.empty((n_paths, n_steps + 1)); S[:, 0] = s0
    V = np.empty((n_paths, n_steps + 1)); V[:, 0] = v0
    for k in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dB = np.sqrt(dt) * z1
        dW = np.sqrt(dt) * (rho * z1 + np.sqrt(1 - rho**2) * z2)
        v = np.maximum(V[:, k], 0.0)
        S[:, k + 1] = S[:, k] * np.exp(-0.5 * v * dt + np.sqrt(v) * dB)
        V[:, k + 1] = V[:, k] + alpha * (b - v) * dt + sigma_v * np.sqrt(v) * dW
    return S, np.maximum(V, 0.0)
