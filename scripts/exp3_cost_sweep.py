"""RESULT 3 — Robustness: how does the edge vary with the cost level?

Sweeps the transaction cost eps every 0.25% from 0.25% to 2%, and at each level
trains several seeds and averages their CVaR-95 improvement over delta hedging.
The seed averaging matters: a single run is noisy, so we report the mean with a
band showing the spread. The expected picture: at tiny costs the two roughly tie
(delta hedging is near-optimal in a frictionless world); the edge GROWS with eps.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import torch
import matplotlib.pyplot as plt
from deephedge.common import (S0, K, SIGMA, T, N, P0_GBM, make_batch_gbm,
                              gbm_features, evaluate)
from deephedge.markets import simulate_gbm
from deephedge.baselines import delta_hedge_gbm
from deephedge.deep import train_hedger
from deephedge.plotting import style, BLUE

# Fewer paths than the headline runs (30k not 50k) because this script trains a model for every
# (cost, seed) pair — a lot of trainings back to back — so we trade a little tail precision for time.
EVAL_PATHS = 30_000

# The cost grid: every 0.25% from 0.25% up to a heavy 2%. Finer than the original four dots so the
# actual shape of the edge-vs-cost curve shows up, not just a handful of widely-spaced points.
COSTS = [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]

# Seeds averaged at each cost level. One seed is noisy — the CVaR objective settles into slightly
# different minima run to run — so we train each cost point across these seeds and report the mean
# plus a ±1 std band. More seeds = smoother, more trustworthy curve, at linear cost in runtime.
SEEDS = [0, 1, 2]


# Retrain a hedger at every (cost, seed) pair, average the CVaR-95 reduction over seeds at each cost,
# and plot the mean against cost with a ±1 std band. steps=3000 matters: CVaR-95 only sees the worst
# 5% of paths, so it converges late and abruptly. 2000 steps clears the cliff at the cheap and expensive
# ends but NOT in the middle (0.75-1.25%), where some seeds were still stuck low and blew up the band —
# 3000 gets every (cost, seed) onto the plateau. Don't drop this below ~2500 or the mid-cost points lie.
def run(outdir, seeds=SEEDS, steps=3000):
    """Sweep the transaction-cost level (seed-averaged) and plot the deep hedger's edge against it.

    Args:
        outdir: directory to save the figure into.
        seeds: list of training seeds to average at every cost level.
        steps: number of training steps per model.

    Returns:
        rows: list of per-cost dicts with the mean/std CVaR-95 reduction and the per-seed values.
    """
    style()
    rows = []
    for eps in COSTS:
        # Fixed eval paths for this cost, shared across every seed so the models are judged on the
        # exact same tape — within a cost level the only thing that varies is the training seed.
        S = simulate_gbm(S0, 0.0, SIGMA, T, N, n_paths=EVAL_PATHS,
                         seed=700_000 + int(eps * 1e4))
        payoff = np.maximum(S[:, -1] - K, 0.0)
        d_bench = delta_hedge_gbm(S, K, T, SIGMA)

        # Train one model per seed and collect the reduction each one achieves. delta_cvar95 is the
        # same every seed (the benchmark doesn't depend on training), so we just keep the last one.
        reds, deep_cvars, delta_cvar95 = [], [], None
        for seed in seeds:
            model, _ = train_hedger(make_batch_gbm(batch=8192), N, n_features=2, loss="cvar",
                                    alpha=0.95, cost_rate=eps, recurrent=True,
                                    steps=steps, seed=seed, log_every=0)
            with torch.no_grad():
                d_deep = model(torch.tensor(gbm_features(S))).numpy()
            m = evaluate(S, d_bench, d_deep, payoff, P0_GBM, eps)
            reds.append(m["cvar95_reduction_pct"])
            deep_cvars.append(m["deep_hedge"]["cvar95"])
            delta_cvar95 = m["delta_hedge"]["cvar95"]

        reds = np.asarray(reds)
        rows.append({"cost": eps,
                     "reduction_mean": float(reds.mean()),
                     "reduction_std": float(reds.std()),
                     "reductions": reds.tolist(),
                     "delta_cvar95": float(delta_cvar95),
                     "deep_cvar95_mean": float(np.mean(deep_cvars))})
        print(f"[sweep] cost {eps:.2%}: reduction {reds.mean():+.1f}% ± {reds.std():.1f}pp  "
              f"(seeds {list(reds.round(1))})")

    # Mean reduction vs cost, with a ±1 std band across seeds and the raw per-seed points on top so
    # the spread is honest — you can see how much of the curve is signal vs run-to-run luck.
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = np.array([r["cost"] * 100 for r in rows])
    mean = np.array([r["reduction_mean"] for r in rows])
    std = np.array([r["reduction_std"] for r in rows])
    ax.fill_between(xs, mean - std, mean + std, color=BLUE, alpha=0.15, label="±1 std across seeds")
    ax.plot(xs, mean, "o-", color=BLUE, lw=2, ms=7, label="mean over seeds")
    for r in rows:
        ax.scatter([r["cost"] * 100] * len(r["reductions"]), r["reductions"],
                   color=BLUE, s=12, alpha=0.35, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("proportional transaction cost (%)")
    ax.set_ylabel("CVaR-95 reduction vs delta hedge (%)")
    ax.set_title("The deep hedger's edge grows with frictions")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figures", "cost_sweep.png"), bbox_inches="tight")
    return rows


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    print(json.dumps(run(outdir), indent=2))
