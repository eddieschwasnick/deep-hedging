"""RESULT 4 — Policy analysis: WHERE does the learned strategy diverge from
delta hedging, and WHY?

Two views, using the trained GBM model from Result 1:
  (left)  learned position vs. spot at mid-life, against the BS delta curve.
          The learned curve is a smoothed / flattened delta — it refuses to
          chase the steep parts because chasing costs money.
  (right) today's TRADE size vs. the gap between current position and BS delta.
          A flat region around zero appears: the no-trade band. The network
          learned "if I'm close enough, don't touch it" without being told.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import torch
import matplotlib.pyplot as plt
from deephedge.common import S0, K, SIGMA, T, N, DT, gbm_features
from deephedge.markets import simulate_gbm, bs_price_delta
from deephedge.baselines import delta_hedge_gbm
from deephedge.deep import HedgerNet
from deephedge.plotting import style, BLUE, ORANGE, GREEN


def run(outdir):
    style()
    model = HedgerNet(N, 2, recurrent=True)
    model.load_state_dict(torch.load(os.path.join(outdir, "model_gbm.pt"),
                                     weights_only=True))
    model.eval()

    # -- view 1: policy vs spot on real simulated paths at mid-life ----------
    S = simulate_gbm(S0, 0.0, SIGMA, T, N, n_paths=20_000, seed=424242)
    with torch.no_grad():
        d_deep = model(torch.tensor(gbm_features(S))).numpy()
    d_bs = delta_hedge_gbm(S, K, T, SIGMA)

    day = 15
    spot = S[:, day]
    order = np.argsort(spot)
    # bin-average the learned policy over spot for a clean curve
    qs = np.linspace(0.005, 0.995, 60)
    edges = np.quantile(spot, qs)
    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (spot >= lo) & (spot < hi)
        if m.sum() > 30:
            centers.append(spot[m].mean()); means.append(d_deep[m, day].mean())
    grid = np.linspace(spot.min(), spot.max(), 200)
    _, bs_curve = bs_price_delta(grid, K, T - day * DT, SIGMA)

    # -- view 2: trade size vs. distance-from-delta (the band) ---------------
    gap = d_bs[:, day] - d_deep[:, day - 1]          # how far off target am I?
    trade = d_deep[:, day] - d_deep[:, day - 1]      # what did the net trade?

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax1.plot(grid, bs_curve, color=ORANGE, lw=2.5, label="BS delta (cost-blind target)")
    ax1.plot(centers, means, "o", color=GREEN, ms=5, label="learned policy (bin means)")
    ax1.axvline(K, color="#bbb", ls="--", lw=1)
    ax1.set_title(f"Policy vs spot, day {day}: a flattened, lazier delta")
    ax1.set_xlabel("spot"); ax1.set_ylabel("position held"); ax1.legend(loc="upper left")

    ax2.scatter(gap, trade, s=3, alpha=0.15, color=BLUE)
    ax2.axhline(0, color="k", lw=0.7); ax2.axvline(0, color="k", lw=0.7)
    lim = np.quantile(np.abs(gap), 0.995)
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
    ax2.plot([-lim, lim], [-lim, lim], color=ORANGE, lw=1.5, ls="--",
             label="trade fully to target (delta-hedge behavior)")
    ax2.set_title("Trade vs. gap-to-delta: the emergent no-trade band")
    ax2.set_xlabel("gap between BS delta and current position")
    ax2.set_ylabel("trade the network chose")
    ax2.legend(loc="upper left")
    ax2.text(0.98, 0.04, "flat around 0 = 'close enough,\ndon't pay to trade'",
             transform=ax2.transAxes, ha="right", fontsize=9.5, color="#444")

    fig.suptitle("Where the learned policy diverges from delta hedging",
                 y=1.02, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figures", "policy_divergence.png"),
                bbox_inches="tight")
    print("saved policy_divergence.png")


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    run(outdir)
