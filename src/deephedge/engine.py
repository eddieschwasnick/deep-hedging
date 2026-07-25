"""The hedging engine + risk measures — the conceptual heart of the project.

`roll_pnl` takes any strategy (a matrix of positions over time) and a price
path, and returns the terminal profit-and-loss for every path. It is the SAME
function for the classical delta hedge and the neural hedge — only the strategy
differs. Understanding this one function is 80% of understanding the paper.
"""
import numpy as np


# The one function that scores every strategy — classical or neural. Give it a matrix of
# positions and a set of price paths and it plays the whole hedge forward: collect the premium,
# bank the trading gains, subtract the transaction costs, pay out the option at the end. The
# only rule is no look-ahead — deltas[:, k] is chosen at t_k, before the move to t_{k+1} lands.
def roll_pnl(S, deltas, payoff, price_charged, cost_rate=0.0):
    """Compute terminal P&L for a batch of paths under a given strategy.

    Args:
        S: (n_paths, n_steps+1) prices of the hedging instrument. For one instrument this is
            2D; the multi-asset case is handled in markets later.
        deltas: (n_paths, n_steps) positions held over each interval. deltas[:, k] is held from
            t_k to t_{k+1} — decided at t_k, BEFORE seeing the move S[:,k+1]-S[:,k]. No look-ahead.
        payoff: (n_paths,) the option liability Z at maturity.
        price_charged: scalar p0, the cash collected upfront.
        cost_rate: proportional transaction cost epsilon (fraction of the dollar amount traded).

    Returns:
        (n_paths,) terminal P&L = -Z + p0 + trading_gains - costs, one number per path.
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
# Smooth risk measure: (1/lam) log E[exp(-lam * PnL)]. It cares about the whole distribution
# but punishes left-tail losses exponentially, so it's differentiable everywhere (no kink) —
# handy as a training loss. lam is the risk aversion; bigger lam means more tail-shy.
def entropic_risk(pnl, lam=1.0):
    """Entropic risk measure of a P&L sample.

    Args:
        pnl: array of terminal P&L, one entry per path.
        lam: risk-aversion parameter; larger values weight the left tail more heavily.

    Returns:
        Scalar "badness" (1/lam) log E[exp(-lam * pnl)]; lower is better.
    """
    pnl = np.asarray(pnl, float)
    # stabilized log-sum-exp
    x = -lam * pnl
    m = x.max()
    return (m + np.log(np.mean(np.exp(x - m)))) / lam


# CVaR / expected shortfall: the average loss over the worst (1-alpha) slice of outcomes. This is
# the number the deep hedger actually minimises, and it's what makes it lean on the tail instead
# of the average. Sort-and-average version here; the differentiable Rockafellar-Uryasev form you
# backprop through lives in deep.cvar_ru.
def cvar(pnl, alpha=0.5):
    """CVaR (expected shortfall) of a P&L sample at level alpha.

    Args:
        pnl: array of terminal P&L, one entry per path.
        alpha: tail level in (0, 1); alpha -> 1 averages a smaller, more extreme tail.

    Returns:
        Scalar loss = mean of the worst (1-alpha) fraction of outcomes, reported as a positive
        number (larger = worse tail).
    """
    pnl = np.asarray(pnl, float)
    losses = -pnl
    q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    return tail.mean() if tail.size else q
