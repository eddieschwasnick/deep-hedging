"""RESULT 5 — Robustness under stochastic vol: the cost sweep from exp3, on Heston paths.

Identical protocol to exp3 (same cost grid, seeds, eval size and step budget) so the two
curves are directly comparable. The only changes are the market: Heston paths, the
(log-moneyness, time-left, V_t) features, the inst-vol delta benchmark and the Monte Carlo
Heston price as the premium collected.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import numpy as np
import torch
from deephedge.common import (S0, K, T, N, HESTON, P0_HESTON, make_batch_heston,
                              heston_features, evaluate, band_benchmark)
from deephedge.markets import simulate_heston
from deephedge.baselines import delta_hedge_heston
from deephedge.deep import train_hedger
from deephedge.plotting import style
from exp3_cost_sweep import COSTS, SEEDS, EVAL_PATHS, plot_sweep


# Same loop as exp3.run: retrain at every (cost, seed) pair, average the CVaR-95 reduction over
# seeds vs both benchmarks, plot mean ± 1 std. steps=3000 for the same late-convergence reason
# as the GBM sweep.
def run(outdir, seeds=SEEDS, steps=3000):
    """Sweep the transaction-cost level on Heston paths and plot the deep hedger's edge against it.

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
        # Fixed eval paths per cost, shared across seeds. Offset from exp3's seeds so the Heston
        # sweep never reuses GBM's random draws.
        S, V = simulate_heston(S0, HESTON["v0"], HESTON["alpha"], HESTON["b"],
                               HESTON["sigma_v"], HESTON["rho"], T, N,
                               n_paths=EVAL_PATHS, seed=800_000 + int(eps * 1e4))
        payoff = np.maximum(S[:, -1] - K, 0.0)
        d_bench = delta_hedge_heston(S, V, K, T)
        S_tune, _ = simulate_heston(S0, HESTON["v0"], HESTON["alpha"], HESTON["b"],
                                    HESTON["sigma_v"], HESTON["rho"], T, N,
                                    n_paths=EVAL_PATHS, seed=850_000 + int(eps * 1e4))
        d_band, h = band_benchmark(S, S_tune, P0_HESTON, eps)
        feats = torch.tensor(heston_features(S, V))

        reds, reds_band, deep_cvars, m = [], [], [], None
        for seed in seeds:
            model, _ = train_hedger(make_batch_heston(batch=8192), N, n_features=3, loss="cvar",
                                    alpha=0.95, cost_rate=eps, recurrent=True,
                                    steps=steps, seed=seed, log_every=0)
            with torch.no_grad():
                d_deep = model(feats).numpy()
            m = evaluate(S, d_bench, d_deep, payoff, P0_HESTON, eps, deltas_band=d_band)
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
        print(f"[sweep-heston] cost {eps:.2%}: vs delta {reds.mean():+.1f}% ± {reds.std():.1f}pp, "
              f"vs band (h={h:.2f}) {reds_band.mean():+.1f}% ± {reds_band.std():.1f}pp", flush=True)

    plot_sweep(rows, "Deep hedger's CVaR-95 edge vs transaction costs (Heston)",
               os.path.join(outdir, "figures", "cost_sweep_heston.png"))
    with open(os.path.join(outdir, "sweep_heston.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(HERE), "results")
    print(json.dumps(run(outdir), indent=2))
