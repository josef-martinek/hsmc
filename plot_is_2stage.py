import os
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_FOLDERS = [
    #"data/run_20260123-190911_",
    "data/run_20260123-191029_",
    "data/run_20260123-191355_",
    "data/run_20260123-191846_",
    "data/run_20260123-194223_",
    #"data/run_20260123-200115_",
    "data/run_20260124-105847_"
]
CI_Z = 1.96   # 95% CI

# -----------------------------
# Helper: compute optimal percent
# -----------------------------
def optimal_percent_with_ci(results_dict, budgets):
    mean_opt = []
    ci_low = []
    ci_high = []

    for budget in budgets:
        percents = sorted(results_dict[budget].keys())
        n_runs = len(results_dict[budget][percents[0]])

        errors = np.array([
            results_dict[budget][p] for p in percents
        ]).T  # (n_runs, n_percents)

        best_idx = np.argmin(errors, axis=1)
        best_percents = np.array(percents)[best_idx]

        mean = np.mean(best_percents)
        std = np.std(best_percents, ddof=1)
        delta = CI_Z * std / np.sqrt(n_runs)

        mean_opt.append(mean)
        ci_low.append(mean - delta)
        ci_high.append(mean + delta)

    return np.array(mean_opt), np.array(ci_low), np.array(ci_high)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 5))

cmap = plt.cm.viridis
colors = cmap(np.linspace(0.2, 0.9, len(RESULTS_FOLDERS)))

for color, RESULTS_FOLDER in zip(colors, RESULTS_FOLDERS):

    RESULTS_FILE = os.path.join(RESULTS_FOLDER, "smc_results.npz")
    data = np.load(RESULTS_FILE, allow_pickle=True)

    W_ex = data["W_ex_by_setup"]
    W_wos = data["W_wos_by_setup"]
    setups = data["setups"]
    N_runs = int(data["n_runs"])

    # ---- Load problem dimension ----
    spec = importlib.util.spec_from_file_location(
        "problem_parameters",
        os.path.join(RESULTS_FOLDER, "problem_parameters.py")
    )
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    param_dim = pp.SamplerConfig.param_dim

    # ---- Organize results ----
    results_ex = defaultdict(lambda: defaultdict(list))
    results_wos = defaultdict(lambda: defaultdict(list))

    for err_ex, err_wos, (budget, percent) in zip(W_ex, W_wos, setups):
        results_ex[budget][percent].append(err_ex)
        results_wos[budget][percent].append(err_wos)

    budgets = sorted(results_ex.keys())

    # ---- Compute stats ----
    mean_ex, lo_ex, hi_ex = optimal_percent_with_ci(results_ex, budgets)
    mean_wos, lo_wos, hi_wos = optimal_percent_with_ci(results_wos, budgets)

    # ---- Plot ----
    plt.errorbar(
        budgets, mean_ex,
        yerr=[mean_ex - lo_ex, hi_ex - mean_ex],
        fmt="o-", capsize=4,
        color=color,
        label=f"Exact (d={param_dim})"
    )

    plt.errorbar(
        budgets, mean_wos,
        yerr=[mean_wos - lo_wos, hi_wos - mean_wos],
        fmt="s--", capsize=4,
        color=color,
        alpha=0.7,
        label=f"WOS (d={param_dim})"
    )

# -----------------------------
# Final plot styling
# -----------------------------
plt.xscale("log")
plt.xlabel("Budget (fine chain steps)", fontsize=13)
plt.ylabel("Optimal fraction on coarse level", fontsize=13)
plt.title("Optimal budget allocation vs total budget", fontsize=14)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()
