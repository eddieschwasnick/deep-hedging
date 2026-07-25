"""Shared pieces for the result scripts: batch factories for training,
feature construction, and a single evaluation routine so every comparison
uses identical out-of-sample data and metrics.
"""
import numpy as np
import torch
from .markets import simulate_gbm, simulate_heston, bs_price_delta
from .engine import roll_pnl, cvar

# Same PROBLEM SETUP ACROSS ALL EXPERIMENTS :
# One 30-day at-the-money call: S0 = K = 100, 20% vol, rebalanced daily (N = 30 steps).
# Pinning this down once here means every script hedges the exact same instrument and the only
# things that change between experiments are the market (GBM vs Heston) and the cost level.
S0, K, SIGMA, T, N = 100.0, 100.0, 0.2, 30 / 365, 30
HESTON = dict(alpha=1.0, b=0.04, sigma_v=2.0, rho=-0.7, v0=0.04)  # Buehler et al. params
DT = T / N
P0_GBM = float(bs_price_delta(S0, K, T, SIGMA)[0])  # fair BS price = cash collected upfront


# The inputs the GBM hedger gets to see each day. Two numbers are enough in a Black-Scholes
# world: how far in/out of the money we are (log-moneyness) and how much time is left. The BS
# delta is a function of exactly these, so the network has everything it needs to match it.
def gbm_features(S):
    """Build the per-step feature array for the GBM hedger.

    Args:
        S: (n_paths, N+1) simulated price paths.

    Returns:
        (n_paths, N, 2) float32 array; last axis is (log-moneyness, time-to-maturity).
    """
    logm = np.log(S[:, :N] / K)
    tleft = (T - np.arange(N) * DT)[None, :].repeat(S.shape[0], axis=0)
    return np.stack([logm, tleft], axis=-1).astype(np.float32)


# Same two features as GBM plus the current variance V_t. Under stochastic vol the right hedge
# depends on where vol is right now so a hedger that can't see V_t is flying blind without that knowledge 
# this extra column is the whole reason the deep hedger can beat an inst-vol delta hedge on Heston paths.
def heston_features(S, V):
    """Build the per-step feature array for the Heston hedger.

    Args:
        S: (n_paths, N+1) simulated price paths.
        V: (n_paths, N+1) instantaneous variance along each path.

    Returns:
        (n_paths, N, 3) float32 array; last axis is (log-moneyness, time-to-maturity, variance).
    """
    logm = np.log(S[:, :N] / K)
    tleft = (T - np.arange(N) * DT)[None, :].repeat(S.shape[0], axis=0)
    return np.stack([logm, tleft, V[:, :N]], axis=-1).astype(np.float32)


# Factory that hands train_hedger a fresh batch of GBM paths every call. We resimulate on each
# step (new seed drawn from the training rng) instead of reusing one fixed set, so the model
# never sees the same paths twice. That's what stops it memorising the noise.
def make_batch_gbm(batch=4096):
    """Make a batch generator for training on GBM paths.

    Args:
        batch: number of paths to simulate per training step.

    Returns:
        A function _mb(rng) -> dict with keys S, feats, payoff, p0, ready to feed train_hedger.
    """
    def _mb(rng):
        seed = int(rng.integers(0, 2**31 - 1))
        S = simulate_gbm(S0, 0.0, SIGMA, T, N, n_paths=batch, seed=seed)
        payoff = np.maximum(S[:, -1] - K, 0.0)
        return {"S": torch.tensor(S, dtype=torch.float32),
                "feats": torch.tensor(gbm_features(S)),
                "payoff": torch.tensor(payoff, dtype=torch.float32),
                "p0": P0_GBM}
    return _mb


# Same idea for Heston. The simulator returns variance too, so we pass it into the features and
# charge the model-free Monte Carlo price (P0_HESTON) upfront rather than a BS price — there's
# no single vol to plug into Black-Scholes here.
def make_batch_heston(batch=4096):
    """Make a batch generator for training on Heston paths.

    Args:
        batch: number of paths to simulate per training step.

    Returns:
        A function _mb(rng) -> dict with keys S, feats, payoff, p0, ready to feed train_hedger.
    """
    def _mb(rng):
        seed = int(rng.integers(0, 2**31 - 1))
        S, V = simulate_heston(S0, HESTON["v0"], HESTON["alpha"], HESTON["b"],
                               HESTON["sigma_v"], HESTON["rho"], T, N,
                               n_paths=batch, seed=seed)
        payoff = np.maximum(S[:, -1] - K, 0.0)
        return {"S": torch.tensor(S, dtype=torch.float32),
                "feats": torch.tensor(heston_features(S, V)),
                "payoff": torch.tensor(payoff, dtype=torch.float32),
                "p0": P0_HESTON}
    return _mb


# There's no closed-form Heston price we trust here, so the risk-neutral price is just a big
# Monte Carlo average of the payoff (drift is already 0 in the simulator). Run once at import
# and cache in P0_HESTON below. It's the cash the Heston hedger collects for writing the call.
# An average.
def _heston_price(n=400_000, seed=12345):
    """Risk-neutral Heston call price by Monte Carlo.

    Args:
        n: number of paths in the pricing simulation.
        seed: rng seed for the pricing paths.

    Returns:
        Scalar price = mean of (S_T - K)^+ under the simulator's risk-neutral dynamics.
    """
    S, _ = simulate_heston(S0, HESTON["v0"], HESTON["alpha"], HESTON["b"],
                           HESTON["sigma_v"], HESTON["rho"], T, N,
                           n_paths=n, seed=seed)
    return float(np.maximum(S[:, -1] - K, 0.0).mean())

P0_HESTON = _heston_price()


# How much trading a strategy does over the life of the hedge, averaged across paths. Pad with a
# flat start and a liquidation at the end (both zero) so opening and closing trades count, then
# sum the absolute position changes. Costs are charged on this activity so lower is cheaper, and is better.
def turnover(deltas):
    """Average total traded quantity per path.

    Args:
        deltas: (n_paths, N) positions held over each interval.

    Returns:
        Mean over paths of sum_k |delta_k - delta_{k-1}|, opening from and closing back to flat.
    """
    pad = np.concatenate([np.zeros((deltas.shape[0], 1)), deltas,
                          np.zeros((deltas.shape[0], 1))], axis=1)
    return float(np.abs(np.diff(pad, axis=1)).sum(axis=1).mean())


# Scores both hedgers on the exact same paths and payoff so the comparison is apples-to-apples.
# Each position matrix goes through the same roll_pnl, and we report the summary stats plus the
# headline number: how much the deep hedger cuts the CVaR-95 tail relative to delta hedging.
def evaluate(S, deltas_bench, deltas_deep, payoff, p0, cost_rate):
    """Score the benchmark and deep hedgers on one shared set of paths.

    Args:
        S: (n_paths, N+1) price paths both strategies are evaluated on.
        deltas_bench: (n_paths, N) benchmark (delta-hedge) positions.
        deltas_deep: (n_paths, N) deep-hedger positions.
        payoff: (n_paths,) option payoff at maturity.
        p0: cash collected upfront (the price charged).
        cost_rate: proportional transaction cost epsilon.

    Returns:
        dict keyed by "delta_hedge" and "deep_hedge" (each with mean_pnl, std_pnl, cvar95,
        cvar99, turnover), plus "cvar95_reduction_pct" — the percentage cut in the CVaR-95 tail.
    """
    out = {}
    for name, d in [("delta_hedge", deltas_bench), ("deep_hedge", deltas_deep)]:
        pnl = roll_pnl(S, d, payoff, p0, cost_rate=cost_rate)
        out[name] = {
            "mean_pnl": float(pnl.mean()),
            "std_pnl": float(pnl.std()),
            "cvar95": float(cvar(pnl, 0.95)),
            "cvar99": float(cvar(pnl, 0.99)),
            "turnover": turnover(d),
        }
    b, dp = out["delta_hedge"]["cvar95"], out["deep_hedge"]["cvar95"]
    out["cvar95_reduction_pct"] = float(100.0 * (b - dp) / abs(b))
    return out
