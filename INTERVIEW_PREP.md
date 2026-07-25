# How a top quant interviewer will attack this project — and how to survive it

Quant interviews about a resume project follow a predictable escalation: they
start friendly, then drill until they find the bottom of your understanding.
The goal isn't to catch you — it's to locate the boundary between "built it and
understands it" and "followed a tutorial." Every question below is one they can
reach directly from your three resume bullets. For each: what they're really
testing, and what a strong answer contains.

---

## Stage 1 — The 90-second opener (testing: can you frame a problem?)

**"Walk me through the project."**

They want a crisp arc, not a feature list. Strong shape: *problem → why the
classical answer fails → your approach → headline number.* Something like:
"Banks hedge sold options by delta hedging, which is provably perfect in a
frictionless market. With transaction costs it's badly suboptimal — it
over-trades, and you can't even define 'the' price anymore. I trained a neural
network to output the position directly, optimizing the 95% CVaR of terminal
hedging P&L through a differentiable market simulation, following Bühler et
al. 2019. Out-of-sample on 50k paths it cut CVaR-95 by X% versus delta hedging
at 1% costs while trading roughly 4× less, and the interesting part is *how*:
it learned a no-trade band without being told one exists."

Red flag they're watching for: leading with architecture ("I used a 2-layer
MLP with ReLU...") instead of the problem. Architecture is Stage 3 material.

**"Why is this interesting? Delta hedging works fine, doesn't it?"**

Testing whether you understand the finance, not the ML. Strong answer hits
three beats: (1) discrete trading already makes hedging imperfect (the 1/√n
error law — you have a test for it); (2) with proportional costs the BS
strategy's turnover is the enemy — following it exactly bleeds money linearly
in trading volume, and superhedging (never lose) is uselessly expensive
(Soner–Shreve–Cvitanić: it costs the whole stock); (3) so hedging becomes a
preference-dependent optimization — someone has to say *how much* tail risk
they'll trade for cost savings, which is exactly what the CVaR objective
encodes.

---

## Stage 2 — The finance drill (testing: do you understand what you optimized?)

**"Define CVaR-95 precisely. Why that and not variance or VaR?"**

Precise definition: the expected loss conditional on being in the worst 5% of
outcomes. Versus variance: variance punishes upside symmetrically with
downside — a hedger doesn't mind windfalls. Versus VaR: VaR is a quantile,
blind to how bad the tail beyond it gets, and it's not coherent
(sub-additivity fails — diversification can look bad under VaR). CVaR is
coherent, regulatorily standard, and — crucially for this project — has a
representation that makes it trainable (next question, guaranteed).

**"CVaR is a tail statistic — sort, take the worst 5%, average. How do you
backpropagate through a sort?"**

This is the single most likely deep question, and your best moment. You don't
backprop through a sort. Rockafellar–Uryasev (2000): CVaR_α(X) = min over w of
{ w + E[(−X − w)⁺]/(1−α) }. The minimization over the scalar w can be *joined*
to the minimization over network weights — w becomes one extra trainable
parameter, the expression inside is a plain expectation of a ReLU, and Monte
Carlo + autograd handles it. Bonus point that lands well: at the optimum, w
converges to the VaR — you can check this in your own runs, and there's a unit
test in the repo asserting the RU minimum equals the sorted-tail CVaR.

**"You charged the Black–Scholes price but hedged under costs. Is that
consistent? What *should* the option cost?"**

Testing whether you know pricing and hedging decoupled. Honest answer: no —
with costs, the BS price undercharges, which is why mean P&L is negative for
*both* strategies in my results; I held the price fixed so the comparison
isolates hedging skill. The right price is the *indifference price*: the p₀
that makes the optimally-hedged seller's risk equal to not trading at all,
p(Z) = π(−Z) − π(0). Extension I know about: as cost ε → 0, the price bump
scales like ε^{2/3} (Whalley–Wilmott), and the deep hedging paper reproduces
that exponent numerically.

**"Why is your mean P&L negative even for the deep hedger?"**

Because both strategies pay real transaction costs out of a premium that
didn't include them. The deep hedger loses less (it trades ~4× less), but any
hedger that trades at all bleeds under a cost-free premium. If I charged the
indifference price instead, the deep hedger's mean would center near zero.

**"What happens to your strategy at expiry, ATM?"** (a favorite trap)

Gamma explodes: delta flips between ~0 and ~1 on tiny moves, so a delta
hedger churns maximally exactly when trading is most expensive per unit of
protection. The learned policy flattens hardest there — it accepts pin risk
rather than pay the churn. If they push: this is the same economics as a
trader "pinning" decision, and it's visible in my policy-vs-spot figure at
late t.

---

## Stage 3 — The ML drill (testing: did you train it or did it train itself?)

**"Describe the architecture and justify every choice."**

One small MLP per trading date (30 of them), two hidden layers, ReLU. Input:
log-moneyness, time-to-maturity (Heston adds instantaneous variance), *plus
the previous position*. Per-step networks because the optimal policy is
time-inhomogeneous — one day to expiry hedges nothing like thirty. The
previous-position input is the load-bearing choice: without it the policy is a
function of market state alone and literally *cannot represent* "I'm close
enough to target, don't trade" — no-trade bands are unexpressible. I verified
this matters empirically: the ablation (memoryless vs recurrent) ties without
costs and the recurrent one wins with costs.

**"What's your loss landscape like? How do you know you converged and not
just stalled?"**

Honest answer: no convergence guarantee exists — the objective is non-convex
in the weights. What I have instead: (1) a problem with a known answer — with
costs off, the trained policy recovers the BS delta curve it was never shown,
which validates the machinery end-to-end; (2) training-noise diagnosis — CVaR-95
means only ~5% of each batch carries gradient, so I raised batch size to 8192
and added cosine LR decay after observing tail metrics lagging mean metrics;
(3) out-of-sample evaluation on disjoint seeds, so any overfitting to training
noise would show up.

**"Where does overfitting even mean anything here? Your simulator is
infinite data."**

Sharp question; the sharp answer: classical overfitting (memorizing samples)
is nearly impossible with fresh batches each step. The real risk relocates to
*model risk*: the network is optimal for my simulator, not for markets. If
true dynamics have jumps, stochastic vol (which is why I added Heston), or
regime shifts, the learned policy inherits none of that robustness. This is
the known critique of deep hedging and the reason the literature moved toward
market-data-driven generators. I'd rather volunteer this limitation than have
it extracted from me.

**"Why RL-flavored language? Is this reinforcement learning?"**

It's policy search, but not model-free RL: because my simulator is
differentiable, I do pathwise gradients straight through the environment — no
Q-function, no Bellman backup, no exploration/exploitation. That's far more
sample-efficient than Q-learning approaches to the same problem (Halperin's
QLBS, Kolm–Ritter). If the environment weren't differentiable (real exchange
fills, discrete lots), I'd need score-function estimators or genuine RL.

**"How would this scale to a book of 1,000 options / 10 underlyings?"**

The framework's selling point: computational cost scales with the number of
*hedging instruments* (network output dim), not the number of liabilities — a
book is just a different payoff Z fed to the same pipeline. Caveats I'd give:
feature engineering grows, per-step nets could share weights with time as an
input, and the paper's own high-dimensional demo (5 Hestons, d=10) trains in
hours on a laptop.

---

## Stage 4 — The engineering & statistics drill (testing: is the number real?)

**"Your X% CVaR reduction — give me the error bar."**

They're testing statistical hygiene. Components: (1) evaluation noise — 50k
paths means the CVaR-95 estimate uses the worst 2,500; I can bootstrap those
for a CI; (2) training noise — retrain across seeds and report the spread
(the honest answer is that seed-to-seed variation is larger than evaluation
noise for tail objectives, which is why single-seed claims are weak). If I
haven't run multi-seed yet, say exactly that and how I'd do it — pretending
robustness you didn't measure is the fastest way to fail this stage.

**"How do you know you don't have look-ahead bias?"**

Position at t_k multiplies the *next* increment S_{k+1} − S_k and is computed
only from information up to t_k. Two guards: code structure (features stop at
column k), and a behavioral test — the √n law. A look-ahead bug makes hedging
error shrink faster than n^{−1/2}; my test asserts the fitted slope lies in
(−0.6, −0.4).

**"What would break if I doubled volatility at test time only?"**

Distribution shift: the network only learned the region its training paths
visit. I've seen this concretely — the learned delta diverges from BS outside
±2σ of the spot distribution. Test-time σ=0.4 against training σ=0.2 would
push most paths into the badly-learned region and performance would degrade
unpredictably. Mitigations: train across a σ range (randomize per batch), or
include realized vol as a feature.

**"Prove the P&L computation is right."**

Point at tests, not vibes: NumPy and torch implementations agree to 1e-8;
costs strictly reduce P&L path-by-path; zero-strategy P&L equals p₀ − Z
exactly; MC price matches the BS closed form; simulator terminal moments match
lognormal closed forms.

---

## Stage 5 — The judgment questions (testing: taste)

**"If you had two more weeks, what's the highest-value addition?"**

Have a ranked answer with reasons. Mine: (1) indifference pricing + the
ε^{2/3} asymptotics plot — it validates the *prices*, not just the hedges,
against independent theory; (2) multi-seed robustness tables — turns every
claim from anecdote to statistic; (3) two-instrument Heston hedging (add the
variance swap) — makes the stochastic-vol result much stronger since vol risk
becomes hedgeable.

**"Would you trade this?"**

The only correct answer is a calibrated no: "Not as-is — it's optimal for a
simulator, and the simulator is wrong in ways I can enumerate. What I *would*
trust is the qualitative structure it discovered: trade less than delta
hedging, run a band, widen it near expiry. That structure matches asymptotic
theory derived independently, which is exactly the kind of agreement that
makes a model useful as a decision aid before it's a decision maker."

**"What's the most surprising thing you learned building it?"**

Have a real one. Strong candidates from this build: that the CVaR-50-trained
hedger had a *worse* 99% tail than delta hedging — the risk measure steers
behavior so literally that optimizing the median tail sacrifices the extreme
tail; or that the no-trade band emerged with no band-shaped anything in the
architecture, purely from the previous-position input plus the cost term.

---

## The three traps, named

1. **The precision trap.** Any number you say ("cut CVaR by 31%") will be
   followed by "how is that measured, exactly?" Never quote a number whose
   denominator, dataset, and seed policy you can't state in one breath.
2. **The attribution trap.** "Did you invent this?" The answer is always: the
   method is Bühler et al. 2019; the implementation, experiments, tests, and
   analysis are mine. Overclaiming novelty ends interviews; precise
   attribution plus demonstrated depth reads as senior.
3. **The limits trap.** Interviewers respect the candidate who volunteers the
   synthetic-data caveat before being asked. The strongest sentence in your
   arsenal is: "Model risk isn't removed here — it's relocated from the
   pricing model to the scenario generator, and I can tell you exactly what
   that costs."
