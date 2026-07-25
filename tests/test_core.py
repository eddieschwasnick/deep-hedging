"""Correctness tests. Sanity checks.
Run: pytest tests/ -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import torch
from deephedge.markets import simulate_gbm, simulate_heston, bs_price_delta
from deephedge.engine import roll_pnl, cvar, entropic_risk
from deephedge.deep import HedgerNet, pnl_torch, cvar_ru


# If the simulator is right, the terminal price is lognormal with known mean and std. Check both
# against the closed form. Broken drift term or vol scaling would show up here.
def test_gbm_matches_lognormal_moments():
    """Simulator sanity check: terminal mean/std match closed form."""
    S = simulate_gbm(100, 0.0, 0.2, 1.0, 252, 200_000, seed=1)
    st = S[:, -1]
    assert abs(st.mean() - 100.0) < 0.15                      # E[S_T] = S0 (mu=0)
    true_std = 100.0 * np.sqrt(np.exp(0.2**2) - 1.0)
    assert abs(st.std() - true_std) / true_std < 0.02


# The BS pricer and the simulator have to agree: Monte Carlo the payoff and it should land on the
# closed-form price. Ties the analytic formula to the paths the hedger actually trades on.
def test_bs_price_against_monte_carlo():
    S = simulate_gbm(100, 0.0, 0.2, 30/365, 30, 400_000, seed=2)
    mc = np.maximum(S[:, -1] - 100.0, 0.0).mean()
    price, _ = bs_price_delta(100.0, 100.0, 30/365, 0.2)
    assert abs(mc - float(price)) < 0.02


# The sharpest correctness check in the suite. A frictionless delta hedge's error shrinks like
# 1/sqrt(n), so on a log-log plot the hedging-error std has slope -1/2. If roll_pnl ever peeked at
# the future move when placing a trade, the error would collapse and this slope would break.
def test_no_lookahead_slope():
    """The sqrt(n) law: hedging-error slope ~ -1/2. A look-ahead bug breaks this."""
    p0 = float(bs_price_delta(100, 100, 30/365, 0.2)[0])
    stds = []
    ns = [10, 40, 160]
    for N in ns:
        S = simulate_gbm(100, 0.0, 0.2, 30/365, N, 30_000, seed=3)
        payoff = np.maximum(S[:, -1] - 100, 0)
        deltas = np.empty((S.shape[0], N))
        for k in range(N):
            _, d = bs_price_delta(S[:, k], 100, 30/365 - k*(30/365)/N, 0.2)
            deltas[:, k] = d
        stds.append(roll_pnl(S, deltas, payoff, p0).std())
    slope = np.polyfit(np.log(ns), np.log(stds), 1)[0]
    assert -0.6 < slope < -0.4


# Costs can only ever hurt: hold the strategy and paths fixed, turn costs on, and every single
# path's P&L must come out no better than the free version. Guards the sign of the cost term.
def test_costs_reduce_pnl_monotonically():
    S = simulate_gbm(100, 0.0, 0.2, 30/365, 30, 5_000, seed=4)
    payoff = np.maximum(S[:, -1] - 100, 0)
    deltas = np.full((5_000, 30), 0.5)
    p_free = roll_pnl(S, deltas, payoff, 2.0, cost_rate=0.0)
    p_cost = roll_pnl(S, deltas, payoff, 2.0, cost_rate=0.01)
    assert np.all(p_cost <= p_free + 1e-9)


# Pin down the two limits of CVaR so the definition can't silently drift: at alpha->0 it averages
# everything (so it's just -mean), and at alpha->1 it's down to the single worst outcome (-min).
def test_cvar_limits():
    """CVaR -> -mean as alpha->0 ; -> -min as alpha->1."""
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1, 100_000)
    assert abs(cvar(pnl, 1e-6) - (-pnl.mean())) < 1e-2
    assert abs(cvar(pnl, 0.99999) - (-pnl.min())) < 0.2


# The numpy scorer (engine.roll_pnl) and the torch scorer (deep.pnl_torch) must compute the exact
# same P&L, or training would optimise a different number than we report. Feed both the same random
# positions and demand they agree to floating-point tolerance.
def test_torch_numpy_pnl_agree():
    S = simulate_gbm(100, 0.0, 0.2, 30/365, 30, 1_000, seed=5)
    payoff = np.maximum(S[:, -1] - 100, 0)
    deltas = np.random.default_rng(1).uniform(0, 1, (1_000, 30))
    a = roll_pnl(S, deltas, payoff, 2.0, cost_rate=0.01)
    b = pnl_torch(torch.tensor(S), torch.tensor(deltas),
                  torch.tensor(payoff), 2.0, cost_rate=0.01).numpy()
    assert np.allclose(a, b, atol=1e-8)


# The differentiable CVaR (cvar_ru) is only worth training on if its minimum over w actually equals
# the true sort-and-average CVaR. Sweep w on a grid, take the min, and check it lands on cvar().
def test_cvar_ru_matches_sorted_cvar_at_optimum():
    """Minimizing the RU form over w recovers the sort-and-average CVaR."""
    rng = np.random.default_rng(2)
    pnl = torch.tensor(rng.normal(0, 1, 50_000))
    ws = torch.linspace(-1, 5, 4001)
    vals = torch.stack([cvar_ru(pnl, w, 0.95) for w in ws])
    assert abs(vals.min().item() - cvar(pnl.numpy(), 0.95)) < 5e-3


# End-to-end training only works if the loss reaches every day's network. Backprop once and confirm
# each per-step net picked up a nonzero gradient — a detached path or broken recurrence would leave
# some day with no signal, and that day would never learn.
def test_gradients_reach_every_step_network():
    """Autograd flows through the whole trading episode into every day's net."""
    torch.manual_seed(0)
    model = HedgerNet(n_steps=10, n_features=2, recurrent=True)
    S = torch.tensor(simulate_gbm(100, 0.0, 0.2, 10/365, 10, 256, seed=6),
                     dtype=torch.float32)
    feats = torch.randn(256, 10, 2)
    payoff = torch.relu(S[:, -1] - 100)
    pnl = pnl_torch(S, model(feats), payoff, 2.0, cost_rate=0.01)
    loss = cvar_ru(pnl, torch.zeros(1, requires_grad=True), 0.95)
    loss.backward()
    for k, net in enumerate(model.nets):
        gnorm = sum(p.grad.abs().sum().item() for p in net.parameters())
        assert gnorm > 0, f"no gradient reached step-{k} network"


# Two things the Heston simulator has to get right: variance never goes negative (the full-truncation
# clip), and it mean-reverts toward b. Starting at v0 = b, the terminal variance should still average b.
def test_heston_variance_stays_nonnegative_and_mean_reverts():
    S, V = simulate_heston(100, 0.04, 1.0, 0.04, 2.0, -0.7, 30/365, 30,
                           50_000, seed=7)
    assert V.min() >= 0.0
    assert abs(V[:, -1].mean() - 0.04) < 0.004   # b = 0.04, starts at b
