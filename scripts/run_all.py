"""
Repoducable script to reproduce every result in the README with a single command:
    python scripts/run_all.py
Writes results/metrics.json and all figures.

Please allow 5-10 min on a laptop CPU.
"""

# Make sure to import the src/ directory so we can run this script from anywhere
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

# Import the experiments in the order they appear in the README. Each experiment
# has a single run() function that takes an output directory and returns a dictionary of metrics. 
# The run() function also saves any figures to the output directory.
import exp1_headline_gbm, exp2_headline_heston, exp3_cost_sweep, exp4_policy_analysis
import exp5_cost_sweep_heston

# Create the output directory and a subdirectory for figures
outdir = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)

# Run each experiment and collect the metrics in a single JSON file
t0 = time.time()
metrics = {}
metrics["gbm_headline"] = exp1_headline_gbm.run(outdir)
metrics["heston_headline"] = exp2_headline_heston.run(outdir)
metrics["cost_sweep"] = exp3_cost_sweep.run(outdir)
exp4_policy_analysis.run(outdir)
metrics["cost_sweep_heston"] = exp5_cost_sweep_heston.run(outdir)

# Save the metrics to a JSON file
with open(os.path.join(outdir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nAll results reproduced in {time.time()-t0:.0f}s -> results/metrics.json")
