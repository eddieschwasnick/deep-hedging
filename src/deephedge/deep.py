"""The deep hedger: per-step MLPs trained end-to-end through the P&L simulation.

Objective = a convex risk measure of terminal P&L (here CVaR-95 via the
Rockafellar-Uryasev representation, or entropic risk). Gradients flow from the
risk measure back through every trading decision.
"""
import numpy as np
import torch
import torch.nn as nn

# Inherit the base class from torch.nn.Module to then customize the forward pass and define the architecture
# of the neural network. The deep hedger is a recurrent neural network that takes in market features and outputs
# trading positions at each time step. The training process involves simulating market paths, calculating terminal
# P&L, and optimizing the network parameters to minimize a chosen risk measure of the P&L distribution.
class HedgerNet(nn.Module):
    """One small feed-forward net per trading date.

    Input at step k: market features (+ previous position if recurrent).
    Output: position to hold over [t_k, t_{k+1}).
    Architecture follows Buehler et al. (2019): 2 hidden layers, ReLU.
    """
    def __init__(self, n_steps, n_features, hidden=None, recurrent=True):
        super().__init__()
        self.n_steps = n_steps
        self.recurrent = recurrent
        in_dim = n_features + (1 if recurrent else 0)
        h = hidden or (in_dim + 15)
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(),
                          nn.Linear(h, h), nn.ReLU(),
                          nn.Linear(h, 1))
            for _ in range(n_steps)
        ])

    def forward(self, feats):
        """feats: (batch, n_steps, n_features) -> deltas (batch, n_steps)."""
        prev = torch.zeros(feats.shape[0], 1, device=feats.device)
        out = []
        for k in range(self.n_steps):
            x = feats[:, k, :]
            if self.recurrent:
                x = torch.cat([x, prev], dim=1)
            d = self.nets[k](x)
            out.append(d)
            prev = d
        return torch.cat(out, dim=1)


def pnl_torch(S, deltas, payoff, p0, cost_rate=0.0):
    """Terminal P&L, differentiable. Mirrors engine.roll_pnl exactly."""
    dS = S[:, 1:] - S[:, :-1]
    gains = (deltas * dS).sum(dim=1)
    zeros = torch.zeros(deltas.shape[0], 1, device=deltas.device)
    padded = torch.cat([zeros, deltas, zeros], dim=1)
    trades = (padded[:, 1:] - padded[:, :-1]).abs()
    costs = cost_rate * (S[:, :trades.shape[1]] * trades).sum(dim=1)
    return -payoff + p0 + gains - costs


def entropic(pnl, lam=1.0):
    return torch.logsumexp(-lam * pnl - np.log(pnl.shape[0]), dim=0) / lam


def cvar_ru(pnl, w, alpha=0.95):
    """CVaR via Rockafellar-Uryasev: min_w  w + E[(-pnl - w)^+]/(1-alpha).
    Differentiable; w is a trained scalar that converges to the VaR."""
    return w + torch.relu(-pnl - w).mean() / (1.0 - alpha)


def train_hedger(make_batch, n_steps, n_features, *, loss="cvar", alpha=0.95, lam=1.0, cost_rate=0.0, 
                 recurrent=True, steps=800, lr=5e-3, seed=0, log_every=100):
    """Train a recurrent deep hedger end-to-end using a convex risk measure of terminal P&L as the loss function.
    
    Args:
        make_batch: callable that returns a batch of simulated paths and payoffs
        n_steps: number of trading dates
        n_features: number of market features per date
        loss: "cvar" or "entropic"
        alpha: CVaR confidence level (ignored if loss="entropic")
        lam: entropic risk parameter (ignored if loss="cvar")
        cost_rate: proportional transaction cost rate
        recurrent: whether to feed previous position as input to next step
        steps: number of training steps
        lr: learning rate for Adam optimizer
        seed: random seed for reproducibility
        log_every: print loss every this many steps

    Returns:
        model: trained HedgerNet
        history: list of loss values over training steps

    Raises:
        ValueError: if loss is not "cvar" or "entropic"

    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Initialize the deep hedger model and parameters for optimization. 
    model = HedgerNet(n_steps, n_features, recurrent=recurrent)
    params = list(model.parameters())

    # Loss is always cvar in our heston and gbm experiments, but keep flexibility to use entropic risk for future experiments.
    # If using cvar, add the single scalar parameter w to this list of parameters to optimize, which converges to the VaR ar 
    # the end of training. This is part of the Rockafellar-Uryasev formulation for CVaR.
    w = torch.zeros(1, requires_grad=True)
    if loss == "cvar":
        params.append(w)
    
    # Set up the optimizer (Adam) and learning rate scheduler (CosineAnnealingLR) for training the model. The optimizer will 
    # update the model parameters to minimize the chosen loss function (in this experiment is cvar).
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 20)

    # The actual training loop
    history = []
    for it in range(steps):
        # make a new batch of simulated paths and payoffs for each step of trainiing.
        b = make_batch(rng)
        # Compute the trading positions (deltas) for the batch using the current model, and then calculate the terminal P&L
        deltas = model(b["feats"])
        pnl = pnl_torch(b["S"], deltas, b["payoff"], b["p0"], cost_rate=cost_rate)

        if loss == "cvar":
            L = cvar_ru(pnl, w, alpha)
        elif loss == "entropic":
            L = entropic(pnl, lam)
        else:
            raise ValueError(f"Unknown loss {loss}")
        
        opt.zero_grad(); L.backward(); opt.step(); sched.step()
        history.append(L.item())
        if log_every and (it % log_every == 0 or it == steps - 1):
            print(f"    step {it:4d}   loss {L.item():8.4f}")
    return model, history
