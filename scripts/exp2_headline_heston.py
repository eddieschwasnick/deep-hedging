"""RESULT 2 — Headline (Heston): stochastic volatility, stock-only hedging.

The benchmark is the honest practitioner move: BS delta evaluated at the
current instantaneous vol sqrt(V_t). The deep hedger sees (log-moneyness,
time-left, V_t) and must cope with vol risk it cannot hedge away — under
transaction costs. Evaluated on 50,000 fresh Heston paths.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import torch
import matplotlib.pyplot as plt
from deephedge.common import (S0, K, T, N, HESTON, P0_HESTON, make_batch_heston,
                              heston_features, evaluate)
from deephedge.markets import simulate_heston
from deephedge.baselines import delta_hedge_heston
from deephedge.deep import train_hedger
from deephedge.plotting import style, ORANGE, GREEN

# Same 1% proportional cost as the GBM headline so the two results are directly comparable. The
# only thing changing between exp1 and exp2 is the market. See exp1 for why 1% (large enough to see).
COST = 0.01

# 50,000 fresh paths to estimate the tail. At CVaR-95 that's 2,500 tail paths behind the metric,
# enough for a stable number without making the script crawl.
EVAL_PATHS = 50_000

# Evaluation seed, held fixed and separate from any training seed so every run scores on the same
# out-of-sample paths. Different from exp1's seed so GBM and Heston aren't secretly reusing paths.
EVAL_SEED = 999_002


# Train the Heston deep hedger, then score it against the inst-vol delta hedge on fresh paths and
# save the P&L histogram plus metrics. Mirrors exp1 step for step, just in the stochastic-vol world.
# steps=3000 for the same reason as the GBM headline: CVaR-95 converges late, so a short budget
# understates the edge. Kept identical to exp1 so the two markets are trained on equal footing.
def run(outdir, seed=0, steps=3000):
    """Run the Heston headline comparison and write the figure and metrics.

    Args:
        outdir: directory to save the model and figures into.
        seed: random seed for training the deep hedger.
        steps: number of training steps for the deep hedger.

    Returns:
        m: dict of metrics (CVaR-95/99 reduction, turnover, mean and std P&L) for both hedgers.
    """
    style()
    print(f"[Heston] training deep hedger  (CVaR-95, cost={COST:.1%}, seed={seed})")

    # Train on Heston batches. n_features=3 here (not 2) because the hedger also sees V_t; the model
    # is saved to results/model_heston.pt. history is discarded — we only want the trained weights.
    model, _ = train_hedger(make_batch_heston(batch=8192), N, n_features=3, loss="cvar",
                            alpha=0.95, cost_rate=COST, recurrent=True,
                            steps=steps, seed=seed)
    torch.save(model.state_dict(), os.path.join(outdir, "model_heston.pt"))

    # Fresh evaluation paths, then both strategies' positions: the benchmark plugs current vol into
    # the BS delta, the deep hedger reads the same (log-moneyness, time-left, V_t) it trained on.
    S, V = simulate_heston(S0, HESTON["v0"], HESTON["alpha"], HESTON["b"],
                           HESTON["sigma_v"], HESTON["rho"], T, N,
                           n_paths=EVAL_PATHS, seed=EVAL_SEED)
    payoff = np.maximum(S[:, -1] - K, 0.0)
    d_bench = delta_hedge_heston(S, V, K, T)
    with torch.no_grad():
        d_deep = model(torch.tensor(heston_features(S, V))).numpy()

    m = evaluate(S, d_bench, d_deep, payoff, P0_HESTON, COST)

    # Roll both P&L distributions again for the histogram (evaluate already has the summary numbers).
    from deephedge.engine import roll_pnl
    pnl_b = roll_pnl(S, d_bench, payoff, P0_HESTON, COST)
    pnl_d = roll_pnl(S, d_deep,  payoff, P0_HESTON, COST)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.linspace(-8, 4, 90)
    ax.hist(pnl_b, bins=bins, color=ORANGE, alpha=0.6, label="delta hedge (inst. vol)")
    ax.hist(pnl_d, bins=bins, color=GREEN, alpha=0.7, label="deep hedge (CVaR-95)")
    ax.axvline(0, color="k", lw=0.8); ax.legend(loc="upper left")
    ax.set_title(f"Heston, {COST:.0%} costs — CVaR-95 cut by {m['cvar95_reduction_pct']:.0f}% "
                 f"({m['delta_hedge']['cvar95']:.2f} → {m['deep_hedge']['cvar95']:.2f})")
    ax.set_xlabel(f"terminal P&L on {EVAL_PATHS:,} out-of-sample paths")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figures", "heston_headline.png"), bbox_inches="tight")
    print(json.dumps(m, indent=2))
    return m


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    run(outdir)
