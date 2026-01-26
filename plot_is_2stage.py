import os
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_FOLDERS = [
    "data/run_20260123-191029_",
    "data/run_20260123-191355_",
    "data/run_20260123-191846_",
    "data/run_20260123-194223_",
    "data/run_20260124-105847_"
]
CI_Z = 1.96   # 95% CI

MARKERS = ["o", "s", "D", "^", "v", "P", "X"]

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
        ]).T

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
# Setup figures
# -----------------------------
fig_ex, ax_ex = plt.subplots(figsize=(7, 5))
fig_wos, ax_wos = plt.subplots(figsize=(7, 5))

cmap = plt.cm.viridis
colors = cmap(np.linspace(0.2, 0.9, len(RESULTS_FOLDERS)))

print("\n===== ERROR SUMMARY =====\n")

# -----------------------------
# Main loop
# -----------------------------
for idx, (color, RESULTS_FOLDER) in enumerate(zip(colors, RESULTS_FOLDERS)):

    RESULTS_FILE = os.path.join(RESULTS_FOLDER, "smc_results.npz")
    data = np.load(RESULTS_FILE, allow_pickle=True)

    W_ex = data["W_ex_by_setup"]
    W_wos = data["W_wos_by_setup"]
    setups = data["setups"]

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

    # ---- Compute stats (UNCHANGED) ----
    mean_ex, lo_ex, hi_ex = optimal_percent_with_ci(results_ex, budgets)
    mean_wos, lo_wos, hi_wos = optimal_percent_with_ci(results_wos, budgets)

    marker = MARKERS[idx % len(MARKERS)]

    # ---- Plot Exact ----
    ax_ex.errorbar(
        budgets, mean_ex,
        yerr=[mean_ex - lo_ex, hi_ex - mean_ex],
        fmt=marker + "-",
        capsize=4,
        color=color,
        label=f"d={param_dim}"
    )

    # ---- Plot WOS ----
    ax_wos.errorbar(
        budgets, mean_wos,
        yerr=[mean_wos - lo_wos, hi_wos - mean_wos],
        fmt=marker + "--",
        capsize=4,
        color=color,
        alpha=0.7,
        label=f"d={param_dim}"
    )

    # ---- Print diagnostics ----
    print(f"Dimension d = {param_dim}")
    for b in budgets:
        ex_vals = np.concatenate(list(results_ex[b].values()))
        wos_vals = np.concatenate(list(results_wos[b].values()))

        print(
            f"  Budget={b:>6} | "
            f"Exact: best={ex_vals.min():.1e}, " #Here replace by mean of minima
            f"fine={np.mean(results_ex[b][0.0]):.1e}, "
            f"worst={ex_vals.max():.1e} | " #Here also replace by mean of maxima
            f"WOS: best={wos_vals.min():.1e}, "
            f"fine={np.mean(results_wos[b][0.0]):.1e}, "
            f"worst={wos_vals.max():.1e}"
        )
    print()

# -----------------------------
# Final styling (unchanged)
# -----------------------------
for ax, title in zip(
    [ax_ex, ax_wos],
    ["Optimal budget allocation (Exact)", "Optimal budget allocation (WOS)"]
):
    ax.set_xscale("log")
    ax.set_xlabel("Budget (fine chain steps)", fontsize=13)
    ax.set_ylabel("Optimal fraction on coarse level", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(ncol=2)

fig_ex.tight_layout()
fig_wos.tight_layout()
plt.show()
