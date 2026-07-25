"""Classical benchmark strategies the deep hedger is measured against.

Both return a (n_paths, n_steps) position matrix — the same interface the
neural hedger produces — so the SAME roll_pnl scores everything.
"""
import numpy as np
from .markets import bs_price_delta


# The textbook benchmark: recompute the BS delta every day from current spot and time left, and
# hold exactly that many shares. It's optimal in a frictionless Black-Scholes world and completely
# blind to transaction costs — which is the point. The deep hedger's whole edge is what it does
# once costs make this cost-blind chasing expensive.
def delta_hedge_gbm(S, K, T, sigma):
    """Black-Scholes delta hedge on GBM paths, recomputed each day.

    Args:
        S: (n_paths, N+1) simulated price paths.
        K: strike.
        T: total time to maturity in years.
        sigma: volatility used in the delta.

    Returns:
        (n_paths, N) matrix of positions — the same interface the deep hedger produces.
    """
    n_paths, n_plus_1 = S.shape
    N = n_plus_1 - 1
    dt = T / N
    deltas = np.empty((n_paths, N))
    for k in range(N):
        _, d = bs_price_delta(S[:, k], K, T - k * dt, sigma)
        deltas[:, k] = d
    return deltas


# The honest Heston benchmark: plug the current instantaneous vol sqrt(V_t) into the BS delta and
# hedge with that. It's exactly what a desk with no stochastic-vol model would do with only the
# stock to trade — so it's a fair fight, not a strawman. The deep hedger gets the same V_t as a
# feature; the question is whether it uses it better than this naive plug-in.
def delta_hedge_heston(S, V, K, T):
    """Practitioner delta hedge on Heston paths using the current instantaneous vol.

    Args:
        S: (n_paths, N+1) simulated price paths.
        V: (n_paths, N+1) instantaneous variance along each path.
        K: strike.
        T: total time to maturity in years.

    Returns:
        (n_paths, N) matrix of positions — BS delta evaluated at sqrt(V_t) each day.
    """
    n_paths, n_plus_1 = S.shape
    N = n_plus_1 - 1
    dt = T / N
    deltas = np.empty((n_paths, N))
    for k in range(N):
        sig = np.sqrt(np.maximum(V[:, k], 1e-8))
        _, d = bs_price_delta(S[:, k], K, T - k * dt, sig)
        deltas[:, k] = d
    return deltas
