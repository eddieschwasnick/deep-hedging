

CVaR-95: is the 95% Conditional Value at Risk, so you basically take the worst case scenario tail bottom 5% of the losses on the P&L and then average it together.
- Does not measure upside, only measures how bad the downside is so we don't blow up accounts

Different from VaR-95 which is just the percentile at 95%, and just single valued. 


Code:

Pytorch:

torch.linspace(-1, 5, 4001) : Makes a single tensor that has 4001 evenly spaced floating point numbers from -1 to 5






File full understanding:

src/ deephedge:
- plotting.py : YES, formats all matplot histograms s they have the same pretty looking style
- __init__.py : YES this is just what tells python you can import the folder deephedge
- markets.py
- engine.py
- deep.py
- common.py
- baselines.py

scripts:
- exp1
- exp2
- exp3
- exp4
- run_all.py

tests:
- test_core.py

results/ figures:
- cost_sweep.png
- gbm_headline.png
- heston_headline.png
- policy_divergence.png

results:
-nmetrics.json
- model_gbm.pt
- model_heston.pt
- sweep_final.json


Doing GBM in powers of 2, such as the batch size for the total number of paths or time steps for a single path is computationaly efficnet for a computers GPU




The difference between the gbm hedger and the heston hedger:

The make_batch data from the make_batch_gbm and the make_batch_heston differ in 3 main ways:
1. The actual simulator simulate_gbm (constant vol) vs simulate_heston (stochastic vol)
2. The number of features is different as gbm_features takes 2 columns (log-moneyness, time-left); while heston_features is 3 columns (log-moneyness, time-left. and current variance V)
3. Price charged is P0_GBM (BS closed form) vs P0_HESTON (risk-neutral price)

The Heston network has a larger input layer (4 vs 3 inputs) to accept the variance feature — and because the hidden width defaults to in_dim + 15, the hidden layers are also one neuron wider.
- This is required or there would be a crash due to a shape mismatch

