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
                              gbm_features, evaluate, band_benchmark)
from deephedge.markets import simulate_gbm
from deephedge.baselines import delta_hedge_gbm
from deephedge.deep import train_hedger
from deephedge.plotting import style, BLUE, ORANGE

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


# Mean reduction vs cost for each benchmark, with a ±1 std band across seeds and the raw per-seed
# points on top so the spread is honest. Two curves: the edge over the cost-blind delta hedge, and the
# edge over the tuned no-trade-band delta hedge (the harder benchmark). Shared with the Heston sweep
# (exp5) so the two figures are drawn identically.
def plot_sweep(rows, title, path):
    """Plot seed-averaged CVaR-95 reduction against cost, vs both benchmarks, and save it.

    Args:
        rows: per-cost dicts as returned by run().
        title: figure title.
        path: file to save the figure to.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = np.array([r["cost"] * 100 for r in rows])
    for key, color, label in [("reduction", BLUE, "vs delta hedge"),
                              ("reduction_vs_band", ORANGE, "vs banded delta hedge")]:
        mean = np.array([r[f"{key}_mean"] for r in rows])
        std = np.array([r[f"{key}_std"] for r in rows])
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)
        ax.plot(xs, mean, "o-", color=color, lw=2, ms=7, label=label)
        for r in rows:
            vals = r["reductions" if key == "reduction" else "reductions_vs_band"]
            ax.scatter([r["cost"] * 100] * len(vals), vals, color=color, s=12, alpha=0.35, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("proportional transaction cost (%)")
    ax.set_ylabel("CVaR-95 reduction (%)")
    ax.set_title(title)
    ax.legend(loc="lower left", title="mean ± 1 std over seeds")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Retrain a hedger at every (cost, seed) pair, average the CVaR-95 reduction over seeds at each cost,
# and plot the mean against cost with a ±1 std band. steps=3000 matters: CVaR-95 only sees the worst
# 5% of paths, so it converges late and abruptly. 2000 steps clears the cliff at the cheap and expensive
# ends but NOT in the middle (0.75-1.25%), where some seeds were still stuck low and blew up the band —
# 3000 gets every (cost, seed) onto the plateau. Don't drop this below ~2500 or the mid-cost points lie.
def run(outdir, seeds=SEEDS, steps=3000):
    """Sweep the transaction-cost level (seed-averaged) and plot the deep hedger's edge against it.

    Args:
        outdir: results directory; the figure and the per-cost rows (JSON) are saved there.
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
        # Banded benchmark, its width tuned for this cost on separate paths. Seed-independent, so
        # computed once per cost like the plain delta hedge.
        S_tune = simulate_gbm(S0, 0.0, SIGMA, T, N, n_paths=EVAL_PATHS,
                              seed=750_000 + int(eps * 1e4))
        d_band, h = band_benchmark(S, S_tune, P0_GBM, eps)

        # Train one model per seed and collect the reduction each one achieves vs both benchmarks.
        # The benchmarks don't depend on training, so their CVaRs are read off the last evaluation.
        reds, reds_band, deep_cvars, m = [], [], [], None
        for seed in seeds:
            model, _ = train_hedger(make_batch_gbm(batch=8192), N, n_features=2, loss="cvar",
                                    alpha=0.95, cost_rate=eps, recurrent=True,
                                    steps=steps, seed=seed, log_every=0)
            with torch.no_grad():
                d_deep = model(torch.tensor(gbm_features(S))).numpy()
            m = evaluate(S, d_bench, d_deep, payoff, P0_GBM, eps, deltas_band=d_band)
            reds.append(m["cvar95_reduction_pct"])
            reds_band.append(m["cvar95_reduction_vs_band_pct"])
            deep_cvars.append(m["deep_hedge"]["cvar95"])

        reds, reds_band = np.asarray(reds), np.asarray(reds_band)
        rows.append({"cost": eps,
                     "reduction_mean": float(reds.mean()),
                     "reduction_std": float(reds.std()),
                     "reductions": reds.tolist(),
                     "reduction_vs_band_mean": float(reds_band.mean()),
                     "reduction_vs_band_std": float(reds_band.std()),
                     "reductions_vs_band": reds_band.tolist(),
                     "band_width": h,
                     "no_hedge_cvar95": m["no_hedge"]["cvar95"],
                     "delta_cvar95": m["delta_hedge"]["cvar95"],
                     "band_cvar95": m["band_hedge"]["cvar95"],
                     "deep_cvar95_mean": float(np.mean(deep_cvars))})
        print(f"[sweep] cost {eps:.2%}: vs delta {reds.mean():+.1f}% ± {reds.std():.1f}pp, "
              f"vs band (h={h:.2f}) {reds_band.mean():+.1f}% ± {reds_band.std():.1f}pp", flush=True)

    plot_sweep(rows, "Deep hedger's CVaR-95 edge vs transaction costs (GBM)",
               os.path.join(outdir, "figures", "cost_sweep.png"))
    with open(os.path.join(outdir, "sweep_gbm.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    print(json.dumps(run(outdir), indent=2))
