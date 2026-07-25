"""Classical benchmark strategies the deep hedger is measured against.

Both return a (n_paths, n_steps) position matrix — the same interface the
neural hedger produces — so the SAME roll_pnl scores everything.
"""
import numpy as np
from .markets import bs_price_delta


def delta_hedge_gbm(S, K, T, sigma):
    """Textbook Black-Scholes delta hedge, recomputed each day from spot and
    time remaining. Ignores transaction costs entirely (that's the point)."""
    n_paths, n_plus_1 = S.shape
    N = n_plus_1 - 1
    dt = T / N
    deltas = np.empty((n_paths, N))
    for k in range(N):
        _, d = bs_price_delta(S[:, k], K, T - k * dt, sigma)
        deltas[:, k] = d
    return deltas


def delta_hedge_heston(S, V, K, T):
    """Practitioner delta hedge in a Heston world: BS delta evaluated with the
    CURRENT instantaneous volatility sqrt(V_t). This is what a desk without a
    stochastic-vol model would actually do — the honest benchmark when only the
    stock is tradable."""
    n_paths, n_plus_1 = S.shape
    N = n_plus_1 - 1
    dt = T / N
    deltas = np.empty((n_paths, N))
    for k in range(N):
        sig = np.sqrt(np.maximum(V[:, k], 1e-8))
        _, d = bs_price_delta(S[:, k], K, T - k * dt, sig)
        deltas[:, k] = d
    return deltas
