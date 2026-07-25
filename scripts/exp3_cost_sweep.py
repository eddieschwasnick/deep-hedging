"""RESULT 3 — Robustness: how does the edge vary with the cost level?

Retrains a deep hedger at each cost level eps in {0.1%, 0.5%, 1%, 2%} and
measures the CVaR-95 improvement over delta hedging on fresh paths each time.
The expected picture: at tiny costs the two roughly tie (delta hedging is
near-optimal in a frictionless world); the deep hedger's edge GROWS with eps.
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

EVAL_PATHS = 30_000
COSTS = [0.001, 0.005, 0.01, 0.02]


def run(outdir, seed=0, steps=1200):
    style()
    rows = []
    for eps in COSTS:
        print(f"[sweep] cost = {eps:.2%}")
        model, _ = train_hedger(make_batch_gbm(batch=8192), N, n_features=2, loss="cvar",
                                alpha=0.95, cost_rate=eps, recurrent=True,
                                steps=steps, seed=seed, log_every=0)
        S = simulate_gbm(S0, 0.0, SIGMA, T, N, n_paths=EVAL_PATHS,
                         seed=700_000 + int(eps * 1e4))
        payoff = np.maximum(S[:, -1] - K, 0.0)
        d_bench = delta_hedge_gbm(S, K, T, SIGMA)
        with torch.no_grad():
            d_deep = model(torch.tensor(gbm_features(S))).numpy()
        m = evaluate(S, d_bench, d_deep, payoff, P0_GBM, eps)
        m["cost"] = eps
        rows.append(m)
        print(f"    CVaR-95: delta {m['delta_hedge']['cvar95']:.3f}  "
              f"deep {m['deep_hedge']['cvar95']:.3f}  "
              f"(-{m['cvar95_reduction_pct']:.0f}%)")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = [r["cost"] * 100 for r in rows]
    ys = [r["cvar95_reduction_pct"] for r in rows]
    ax.plot(xs, ys, "o-", color=BLUE, lw=2, ms=8)
    ax.set_xlabel("proportional transaction cost (%)")
    ax.set_ylabel("CVaR-95 reduction vs delta hedge (%)")
    ax.set_title("The deep hedger's edge grows with frictions")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figures", "cost_sweep.png"), bbox_inches="tight")
    return rows


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    print(json.dumps(run(outdir), indent=2))
