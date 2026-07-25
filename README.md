# Deep Hedging of European Options Under Transaction Costs

**Python · PyTorch · NumPy** — a neural network learns dynamic hedging strategies
for European calls under proportional transaction costs, trained end-to-end by
backpropagating a tail-risk objective (95% CVaR) through a differentiable
P&L simulation. Benchmarked against Black–Scholes delta hedging on out-of-sample
GBM and Heston (stochastic-volatility) paths.

Reference: Bühler, Gonon, Teichmann & Wood, *Deep Hedging*, Quantitative Finance 19(8), 2019.

## Headline results

Evaluated on **50,000 out-of-sample paths per market** (training never sees these
seeds), 30-day at-the-money European call, daily rebalancing, 1% proportional
transaction costs, CVaR-95 training objective.

| Market | Metric | Delta hedge | Deep hedge | Change |
|---|---|---:|---:|---:|
| GBM | 95% CVaR of hedging error | 4.74 | 4.44 | **−6.33%** |
| GBM | mean P&L | -2.74 | -1.35 | |
| GBM | turnover (shares traded) | 2.72 | 1.36 | |
| Heston | 95% CVaR of hedging error | 6.50 | 5.16 | **−20.60%** |
| Heston | mean P&L | -2.38 | -0.68 | |
| Heston | turnover | 2.36 | 0.67 | |

![GBM headline](results/figures/gbm_headline.png)
![Heston headline](results/figures/heston_headline.png)

## Robustness across cost levels

The deep hedger's edge over delta hedging as a function of the proportional
cost ε (each point is a freshly trained model, evaluated on 30,000 fresh paths):

![Cost sweep](results/figures/cost_sweep.png)

The edge is positive at every cost level tested (9–24% CVaR-95 reduction).
The non-monotonic wiggle across levels is single-seed training variance — tail
objectives are noisy because only ~5% of each batch carries gradient — which is
why the next step listed under limitations is multi-seed reporting. At
near-zero costs the edge is smallest, as expected: delta hedging is close to
optimal in a frictionless market, so there is little left to gain.

## Where (and why) the learned policy diverges from delta hedging

![Policy divergence](results/figures/policy_divergence.png)

Two mechanisms, both cost-driven and both emergent (never programmed in):

1. **A flattened delta.** The learned position-vs-spot curve is a smoothed,
   "lazier" version of the BS delta — it declines to chase the steep gamma
   region near the strike because chasing costs money every day.
2. **A no-trade band.** Plotting the network's chosen trade against its current
   distance from the BS delta reveals a flat region around zero: when the gap
   is small, the optimal action is to do nothing. This reproduces the classical
   Whalley–Wilmott asymptotic result — discovered here from raw P&L alone.

## Repository layout

```
src/deephedge/
  markets.py     GBM + Heston simulators, Black–Scholes closed forms
  baselines.py   delta-hedge benchmarks (GBM; Heston w/ instantaneous vol)
  engine.py      roll_pnl (strategy -> P&L distribution), CVaR, entropic risk
  deep.py        HedgerNet (per-step MLPs, recurrent in previous position),
                 differentiable CVaR (Rockafellar–Uryasev), training loop
  common.py      canonical problem setup, batch factories, unified evaluation
scripts/
  exp1_headline_gbm.py      train + evaluate on GBM        (Result 1)
  exp2_headline_heston.py   train + evaluate on Heston     (Result 2)
  exp3_cost_sweep.py        robustness across cost levels  (Result 3)
  exp4_policy_analysis.py   policy-divergence figures      (Result 4)
  run_all.py                reproduce everything -> results/metrics.json
tests/test_core.py          9 correctness tests (see below)
results/                    metrics.json, trained models, figures
```

## Reproduce

```bash
pip install -r requirements.txt
pytest tests/ -q            # 9 tests, ~45 s
python scripts/run_all.py   # all results + figures, ~15-20 min on laptop CPU
```

All randomness is seeded; evaluation seeds are disjoint from training seeds.

## Method in one paragraph

The strategy is a sequence of small MLPs, one per trading day; day *k*'s network
maps (log-moneyness, time-to-maturity[, instantaneous variance]) **plus its own
previous position** to the position held over the next interval. Feeding back
the previous position is what lets a no-trade band exist — without it the
policy cannot express "I'm close enough, don't trade." A batch of simulated
paths is rolled through the policy to terminal P&L (option payoff, trading
gains, proportional costs); the loss is the CVaR-95 of that P&L written in the
Rockafellar–Uryasev form `w + E[(-PnL - w)+]/(1-α)` with `w` a learned scalar,
which makes the tail risk differentiable. Adam with cosine LR decay,
batch 8192, 1,500 steps; the simulator provides effectively infinite training
data, so evaluation happens exclusively on held-out seeds.

## What the tests guard

Simulator moments vs. closed form · BS price vs. Monte Carlo · the √n
hedging-error law (a look-ahead-bias detector) · costs strictly reduce P&L ·
CVaR limiting behavior (α→0 mean, α→1 worst case) · NumPy/torch P&L parity ·
the RU minimization recovers sorted-tail CVaR · gradients reach every
time-step's network · Heston variance nonnegative + mean-reverting.

## Honest limitations

Synthetic markets (the simulator is the ground truth — model risk is moved to
the scenario generator, not removed); one option, one maturity per trained
model; proportional costs only (no market impact); daily rebalancing. Natural
extensions: CVaR-99 objective, multi-instrument Heston hedging with a variance
swap, market-data-driven scenario generation.
