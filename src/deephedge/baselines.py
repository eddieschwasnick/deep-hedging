"""Classical benchmark strategies the deep hedger is measured against.

Every strategy returns a (n_paths, n_steps) position matrix — the same interface the
neural hedger produces — so the SAME roll_pnl scores everything.
"""
import numpy as np
from .markets import bs_price_delta
from .engine import roll_pnl, cvar


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


# Candidate half-widths for the no-trade band, in shares. Runs past where tuning ever lands
# (≈0.3 at 1-2% costs) so the chosen width is never pinned at the edge of the grid.
BAND_WIDTHS = np.round(np.arange(0.0, 0.62, 0.02), 2)


# The cost-aware classical benchmark: follow a target delta, but only trade when the position has
# drifted more than h shares away from it, and then only back to the nearest edge of the band
# (Whalley-Wilmott style). h=0 is the plain delta hedge. Without a band a cost-blind delta hedge can
# do worse than not hedging at all at high costs, which would flatter the deep hedger.
def band_hedge(target, h):
    """Delta hedge with a no-trade band of half-width h around a target position.

    Args:
        target: (n_paths, N) target positions, e.g. a BS delta hedge.
        h: band half-width in shares; 0 reproduces the target exactly.

    Returns:
        (n_paths, N) positions. Starts flat, and at each step moves only as far as needed to get
        back inside [target - h, target + h].
    """
    pos = np.empty_like(target)
    cur = np.zeros(target.shape[0])
    for k in range(target.shape[1]):
        cur = np.clip(cur, target[:, k] - h, target[:, k] + h)
        pos[:, k] = cur
    return pos


# Pick the band width that minimises CVaR-95 for a given cost. Tuned on its own paths (never the
# evaluation paths), so the benchmark gets the same out-of-sample treatment as the deep hedger.
def tune_band_width(target, S, payoff, p0, cost_rate, widths=BAND_WIDTHS):
    """Choose the no-trade band half-width with the lowest CVaR-95 on tuning paths.

    Args:
        target: (n_paths, N) target positions on the tuning paths.
        S: (n_paths, N+1) tuning price paths.
        payoff: (n_paths,) option payoff on the tuning paths.
        p0: cash collected upfront.
        cost_rate: proportional transaction cost epsilon.
        widths: candidate half-widths to search.

    Returns:
        The best half-width h (float).
    """
    scores = [cvar(roll_pnl(S, band_hedge(target, h), payoff, p0, cost_rate), 0.95) for h in widths]
    return float(widths[int(np.argmin(scores))])
