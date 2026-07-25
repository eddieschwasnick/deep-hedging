"""Shared pieces for the result scripts: batch factories for training,
feature construction, and a single evaluation routine so every comparison
uses identical out-of-sample data and metrics.
"""
import numpy as np
import torch
from .markets import simulate_gbm, simulate_heston, bs_price_delta
from .engine import roll_pnl, cvar

# ---- canonical problem setup (same across all experiments) ----
S0, K, SIGMA, T, N = 100.0, 100.0, 0.2, 30 / 365, 30
HESTON = dict(alpha=1.0, b=0.04, sigma_v=2.0, rho=-0.7, v0=0.04)  # Buehler et al. params
DT = T / N
P0_GBM = float(bs_price_delta(S0, K, T, SIGMA)[0])


def gbm_features(S):
    """Features per day: log-moneyness and time-to-maturity."""
    logm = np.log(S[:, :N] / K)
    tleft = (T - np.arange(N) * DT)[None, :].repeat(S.shape[0], axis=0)
    return np.stack([logm, tleft], axis=-1).astype(np.float32)


def heston_features(S, V):
    """Heston adds the current variance — the state a vol-aware hedger needs."""
    logm = np.log(S[:, :N] / K)
    tleft = (T - np.arange(N) * DT)[None, :].repeat(S.shape[0], axis=0)
    return np.stack([logm, tleft, V[:, :N]], axis=-1).astype(np.float32)


def make_batch_gbm(batch=4096):
    def _mb(rng):
        seed = int(rng.integers(0, 2**31 - 1))
        S = simulate_gbm(S0, 0.0, SIGMA, T, N, n_paths=batch, seed=seed)
        payoff = np.maximum(S[:, -1] - K, 0.0)
        return {"S": torch.tensor(S, dtype=torch.float32),
                "feats": torch.tensor(gbm_features(S)),
                "payoff": torch.tensor(payoff, dtype=torch.float32),
                "p0": P0_GBM}
    return _mb


def make_batch_heston(batch=4096):
    def _mb(rng):
        seed = int(rng.integers(0, 2**31 - 1))
        S, V = simulate_heston(S0, HESTON["v0"], HESTON["alpha"], HESTON["b"],
                               HESTON["sigma_v"], HESTON["rho"], T, N,
                               n_paths=batch, seed=seed)
        payoff = np.maximum(S[:, -1] - K, 0.0)
        # charge the model-free Monte Carlo price (risk-neutral expectation)
        return {"S": torch.tensor(S, dtype=torch.float32),
                "feats": torch.tensor(heston_features(S, V)),
                "payoff": torch.tensor(payoff, dtype=torch.float32),
                "p0": P0_HESTON}
    return _mb


# risk-neutral Heston call price by big Monte Carlo (computed once, cached)
def _heston_price(n=400_000, seed=12345):
    S, _ = simulate_heston(S0, HESTON["v0"], HESTON["alpha"], HESTON["b"],
                           HESTON["sigma_v"], HESTON["rho"], T, N,
                           n_paths=n, seed=seed)
    return float(np.maximum(S[:, -1] - K, 0.0).mean())

P0_HESTON = _heston_price()


def turnover(deltas):
    pad = np.concatenate([np.zeros((deltas.shape[0], 1)), deltas,
                          np.zeros((deltas.shape[0], 1))], axis=1)
    return float(np.abs(np.diff(pad, axis=1)).sum(axis=1).mean())


def evaluate(S, deltas_bench, deltas_deep, payoff, p0, cost_rate):
    """One evaluation to rule them all: same paths, same metrics, both hedgers."""
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
