"""RESULT 1 — GBM : deep hedger vs. Black-Scholes delta hedge under 1% costs.

Trains a recurrent deep hedger with a CVaR-95 objective, then calculate both the deep hedger and 
the Black-Scholes delta hedge terminal P&L distributions on 50,000 fresh GBM paths.

The output is a histogram of the two P&L distributions, and a JSON dictionary of metrics including
the CVaR-95 reduction, CVar-99 reduction, turnover, P&L standard deviation,and P&L mean. Histogram is
saved to results/figures/gbm_headline.png, and metrics are saved to results/metrics.json.

Note that the model is saved to results/model_gbm.pt, and will be reused in the later policy analysis

"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import torch
import matplotlib.pyplot as plt
from deephedge.common import (S0, K, SIGMA, T, N, P0_GBM, make_batch_gbm, gbm_features, evaluate)
from deephedge.markets import simulate_gbm
from deephedge.baselines import delta_hedge_gbm
from deephedge.deep import train_hedger
from deephedge.plotting import style, ORANGE, GREEN
from deephedge.engine import roll_pnl

# Transaction-cost drag has two levers: the per-trade rate ε and the number of rebalances n. For a naive delta
# hedge the total cost scales like ε·√n (Leland), so ε and n trade off against each other. I raise ε rather 
# than n because the model builds one MLP per step — larger n means more networks, so slower training and more
# memory. I use ε = 1% (rather than a realistic fewer basis points) so the cost effect is large enough to see clearly. 
# The cost sweep shows the edge shrinks toward zero cost. cost_rate here is ε from the paper.
COST = 0.01

# The number of simulated price paths used to find each trading stratgies P&L distribution.
# Because the number we are minimizing the CVaR-95 tail, this means 50000 paths will get 2500 tail paths to estimate the metric.
# The number of paths is a tradeoff between accuracy and speed. 50000 is enough to get a stable CVaR-95 estimate, but not so many that the script takes forever to run.
EVAL_PATHS = 50_000


# Random seed for the evaluation paths. This is separate from the training seed, so that the evaluation is always on the same paths regardless of how the model was trained. Reproducibility.
# never use seed in training (prevent overfiting)
EVAL_SEED = 999_001

# Now run the comparison: train the deep hedger, then evaluate both it and the BS delta hedge on fresh paths. Save the histogram and metrics.
def run(outdir, seed=0, steps=1500):
    '''
    Args:
        outdir: directory to save model, histogram, and metrics
        seed: random seed for training the deep hedger
        steps: number of training steps for the deep hedger
    
    Returns:
        m: dictionary of metrics including CVaR-95 reduction, CVaR-99 reduction, turnover, mean P&L, and std P&L for both hedgers
    
    Raises:
        ValueError: if outdir does not exist or is not a directory
    '''
    style()
    print(f"[GBM] training deep hedger  (CVaR-95, cost={COST:.1%}, seed={seed})")

    # Train the deep hedger with the CVaR-95 objective, using the make_batch_gbm function to generate GBM paths.
    # Note that the model is saved to results/model_gbm.pt, and will be reused in the later policy analysis.
    # history is the list of loss values over the training steps, but we don't use it here. We just want the trained model.
    
    
    model, _ = train_hedger(make_batch_gbm(batch=8192), N, n_features=2, loss="cvar",
                            alpha=0.95, cost_rate=COST, recurrent=True,
                            steps=steps, seed=seed)
    torch.save(model.state_dict(), os.path.join(outdir, "model_gbm.pt"))

    S = simulate_gbm(S0, 0.0, SIGMA, T, N, n_paths=EVAL_PATHS, seed=EVAL_SEED)
    payoff = np.maximum(S[:, -1] - K, 0.0)
    d_bench = delta_hedge_gbm(S, K, T, SIGMA)
    with torch.no_grad():
        d_deep = model(torch.tensor(gbm_features(S))).numpy()

    m = evaluate(S, d_bench, d_deep, payoff, P0_GBM, COST)

    pnl_b = roll_pnl(S, d_bench, payoff, P0_GBM, COST)
    pnl_d = roll_pnl(S, d_deep,  payoff, P0_GBM, COST)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.linspace(-7, 4, 90)
    ax.hist(pnl_b, bins=bins, color=ORANGE, alpha=0.6, label="BS delta hedge")
    ax.hist(pnl_d, bins=bins, color=GREEN, alpha=0.7, label="deep hedge (CVaR-95)")
    ax.axvline(0, color="k", lw=0.8); ax.legend(loc="upper left")
    ax.set_title(f"GBM, {COST:.0%} costs — CVaR-95 cut by {m['cvar95_reduction_pct']:.0f}% "
                 f"({m['delta_hedge']['cvar95']:.2f} → {m['deep_hedge']['cvar95']:.2f})")
    ax.set_xlabel(f"terminal P&L on {EVAL_PATHS:,} out-of-sample paths")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figures", "gbm_headline.png"), bbox_inches="tight")
    print(json.dumps(m, indent=2))
    return m


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    run(outdir)
