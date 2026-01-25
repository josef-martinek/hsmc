import os
import numpy as np
import matplotlib.pyplot as plt
import json
import importlib.util
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_FOLDER = "data/run_20260123-190911_"
RESULTS_FILE = os.path.join(RESULTS_FOLDER, "smc_results.npz")
CI_Z = 1.96   # 95% CI

# -----------------------------
# Load results
# -----------------------------
data = np.load(RESULTS_FILE, allow_pickle=True)

W_ex = data["W_ex_by_setup"]
W_wos = data["W_wos_by_setup"]
setups = data["setups"]   # each entry: [budget, percent]
N_runs = int(data["n_runs"])

# -----------------------------
# Load problem dimension
# -----------------------------
spec = importlib.util.spec_from_file_location(
    "problem_parameters",
    os.path.join(RESULTS_FOLDER, "problem_parameters.py")
)
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)
param_dim = pp.SamplerConfig.param_dim

# -----------------------------
# Organize results by (budget, percent)
# -----------------------------
# structure:
# results[budget][percent] = list of errors (length N_runs)
results_ex = defaultdict(lambda: defaultdict(list))
results_wos = defaultdict(lambda: defaultdict(list))

for err_ex, err_wos, (budget, percent) in zip(W_ex, W_wos, setups):
    results_ex[budget][percent].append(err_ex)
    results_wos[budget][percent].append(err_wos)

budgets = sorted(results_ex.keys())

# -----------------------------
# Compute optimal percent per run
# -----------------------------
def optimal_percent_with_ci(results_dict):
    """
    For each budget:
      - find per-run optimal percent
      - compute mean and CI across runs
    """
    mean_opt = []
    ci_low = []
    ci_high = []

    for budget in budgets:
        percents = sorted(results_dict[budget].keys())
        n_runs = len(results_dict[budget][percents[0]])

        # shape: (n_runs, n_percents)
        errors = np.array([
            results_dict[budget][p] for p in percents
        ]).T

        # per-run argmin
        best_idx = np.argmin(errors, axis=1)
        best_percents = np.array(percents)[best_idx]

        mean = np.mean(best_percents)
        std = np.std(best_percents, ddof=1)

        delta = CI_Z * std / np.sqrt(n_runs)

        mean_opt.append(mean)
        ci_low.append(mean - delta)
        ci_high.append(mean + delta)

    return np.array(mean_opt), np.array(ci_low), np.array(ci_high)

mean_ex, lo_ex, hi_ex = optimal_percent_with_ci(results_ex)
mean_wos, lo_wos, hi_wos = optimal_percent_with_ci(results_wos)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 5))

cmap = plt.cm.viridis
color = cmap(0.6)  # one dimension now; extendable later

plt.errorbar(
    budgets, mean_ex,
    yerr=[mean_ex - lo_ex, hi_ex - mean_ex],
    fmt="o-", capsize=4,
    label=f"Exact posterior (d={param_dim})",
    color=color
)

plt.errorbar(
    budgets, mean_wos,
    yerr=[mean_wos - lo_wos, hi_wos - mean_wos],
    fmt="s--", capsize=4,
    label=f"WOS posterior (d={param_dim})",
    color=color, alpha=0.7
)

plt.xscale("log")
plt.xlabel("Budget (fine chain steps)", fontsize=13)
plt.ylabel("Optimal fraction on coarse level", fontsize=13)
plt.title("Optimal budget allocation vs total budget", fontsize=14)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_FOLDER, "optimal_budget_allocation.png"))
plt.show()
