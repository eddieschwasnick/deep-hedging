"""The hedging engine + risk measures — the conceptual heart of the project.

`roll_pnl` takes any strategy (a matrix of positions over time) and a price
path, and returns the terminal profit-and-loss for every path. It is the SAME
function for the classical delta hedge and the neural hedge — only the strategy
differs. Understanding this one function is 80% of understanding the paper.
"""
import numpy as np


def roll_pnl(S, deltas, payoff, price_charged, cost_rate=0.0):
    """Compute terminal P&L for a batch of paths under a given strategy.

    Parameters
    ----------
    S            : (n_paths, n_steps+1) prices of the hedging instrument(s).
                   For one instrument, 2D. (Multi-asset handled in markets later.)
    deltas       : (n_paths, n_steps) positions held over each interval.
                   deltas[:, k] is held from t_k to t_{k+1}  <-- decided at t_k,
                   BEFORE seeing the move S[:,k+1]-S[:,k]. (No look-ahead!)
    payoff       : (n_paths,) the option liability Z at maturity.
    price_charged: scalar p0, the cash collected upfront.
    cost_rate    : proportional transaction cost epsilon (fraction of traded $).

    Returns
    -------
    pnl : (n_paths,)  terminal P&L  =  -Z + p0 + trading_gains - costs
    """
    S = np.asarray(S, float)
    deltas = np.asarray(deltas, float)

    # Trading gains: position * price move, summed over the path.
    dS = S[:, 1:] - S[:, :-1]                 # (n_paths, n_steps)
    trading_gains = np.sum(deltas * dS, axis=1)

    # Transaction costs: pay cost_rate * price * |change in position| at each trade.
    # Start flat (delta_{-1}=0) and liquidate at the end (delta_n=0).
    prev = np.zeros((deltas.shape[0], 1))
    padded = np.concatenate([prev, deltas, np.zeros((deltas.shape[0], 1))], axis=1)
    trades = np.abs(np.diff(padded, axis=1))            # |delta_k - delta_{k-1}|
    trade_prices = S[:, :trades.shape[1]]
    costs = cost_rate * np.sum(trade_prices * trades, axis=1)

    return -payoff + price_charged + trading_gains - costs


# ---------------------------------------------------------------------------
# Risk measures — how you score a whole P&L distribution with one number.
# Convention: these return "badness". Lower = better. (rho in the paper.)
# ---------------------------------------------------------------------------
def entropic_risk(pnl, lam=1.0):
    """Entropic risk: (1/lam) log E[exp(-lam * PnL)]. Smooth; loves the mean
    but punishes left-tail losses exponentially. lam = risk aversion."""
    pnl = np.asarray(pnl, float)
    # stabilized log-sum-exp
    x = -lam * pnl
    m = x.max()
    return (m + np.log(np.mean(np.exp(x - m)))) / lam


def cvar(pnl, alpha=0.5):
    """CVaR / expected shortfall at level alpha: the average of the worst
    (1-alpha) fraction of outcomes (as a positive 'loss'). alpha->1 = cautious.

    Uses the sort-and-average definition. The Rockafellar-Uryasev minimization
    form (the one you backprop through) lives in dh/torch_risk.py.
    """
    pnl = np.asarray(pnl, float)
    losses = -pnl
    q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    return tail.mean() if tail.size else q
