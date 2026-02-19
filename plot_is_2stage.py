import os
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from collections import defaultdict
# Currently what I'm doing in this script is for each budget 
# find the optimal resource allocation. Do this for each run 
# # and then average over the minima. One could also do it the 
# # other way around, first average over the runs and from this 
# # get the minimum

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_FOLDERS = [
    "data/run_20260130-103339_",
    "data/run_20260130-103416_",
    "data/run_20260130-103448_",
    "data/run_20260130-103520_",
    "data/run_20260130-103719_",
    "data/run_20260130-103745_"
]

CI_Z = 1.96   # 95% CI
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]

# >>> NEW: choose dimension for error-vs-budget plot
PLOT_SINGLE_DIM = 64

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

# >>> NEW: error vs budget plot
fig_err, ax_err = plt.subplots(figsize=(7, 5))

cmap = plt.cm.viridis
colors = cmap(np.linspace(0.2, 0.9, len(RESULTS_FOLDERS)))

# -----------------------------
# Error summary buffer
# -----------------------------
error_summary_lines = []
error_summary_lines.append("===== ERROR SUMMARY =====\n")
error_summary_lines.append("Results folders used:\n")
for rf in RESULTS_FOLDERS:
    error_summary_lines.append(f"  - {rf}")
error_summary_lines.append("")

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

    # ---- Compute optimal allocations ----
    mean_ex, lo_ex, hi_ex = optimal_percent_with_ci(results_ex, budgets)
    mean_wos, lo_wos, hi_wos = optimal_percent_with_ci(results_wos, budgets)

    marker = MARKERS[idx % len(MARKERS)]

    # ---- Plot Exact optimal allocation ----
    ax_ex.errorbar(
        budgets, mean_ex,
        yerr=[mean_ex - lo_ex, hi_ex - mean_ex],
        fmt=marker + "-",
        capsize=4,
        color=color,
        label=f"d={param_dim}"
    )

    # ---- Plot WOS optimal allocation ----
    ax_wos.errorbar(
        budgets, mean_wos,
        yerr=[mean_wos - lo_wos, hi_wos - mean_wos],
        fmt=marker + "--",
        capsize=4,
        color=color,
        alpha=0.7,
        label=f"d={param_dim}"
    )

    # ---- Diagnostics ----
    header = f"Dimension d = {param_dim}"
    print(header)
    error_summary_lines.append(header)

    for b in budgets:

        percents_ex = sorted(results_ex[b].keys())
        errors_ex = np.array([
            results_ex[b][p] for p in percents_ex
        ]).T

        best_ex_mean  = np.mean(np.min(errors_ex, axis=1))
        worst_ex_mean = np.mean(np.max(errors_ex, axis=1))
        fine_ex_mean  = np.mean(results_ex[b][0.0])

        percents_wos = sorted(results_wos[b].keys())
        errors_wos = np.array([
            results_wos[b][p] for p in percents_wos
        ]).T

        best_wos_mean  = np.mean(np.min(errors_wos, axis=1))
        worst_wos_mean = np.mean(np.max(errors_wos, axis=1))
        fine_wos_mean  = np.mean(results_wos[b][0.0])

        line = (
            f"  Budget={b:>6} | "
            f"Exact: best={best_ex_mean:.1e}, "
            f"fine={fine_ex_mean:.1e}, "
            f"worst={worst_ex_mean:.1e} | "
            f"WOS: best={best_wos_mean:.1e}, "
            f"fine={fine_wos_mean:.1e}, "
            f"worst={worst_wos_mean:.1e}"
        )

        print(line)
        error_summary_lines.append(line)

    error_summary_lines.append("")
    print()

    # >>> NEW: Exact error vs budget plot for selected dimension
    if param_dim == PLOT_SINGLE_DIM:

        err_means = []
        err_lows = []
        err_highs = []

        err_fo_means = []
        err_fo_lows = []
        err_fo_highs = []

        for b in budgets:
            percents_ex = sorted(results_ex[b].keys())
            errors_ex = np.array([
                results_ex[b][p] for p in percents_ex
            ]).T

            best_errors = np.min(errors_ex, axis=1)

            fine_only_mean  = np.mean(results_ex[b][0.0])
            mean = np.mean(best_errors)
            fine_only_std = np.std(results_ex[b][0.0], ddof=1)
            std = np.std(best_errors, ddof=1)
            fine_only_delta = CI_Z * fine_only_std / np.sqrt(len(results_ex[b][0.0]))
            delta = CI_Z * std / np.sqrt(len(best_errors))

            err_fo_means.append(fine_only_mean)
            err_means.append(mean)
            err_fo_lows.append(fine_only_mean - fine_only_delta)
            err_lows.append(mean - delta)
            err_fo_highs.append(fine_only_mean + fine_only_delta)
            err_highs.append(mean + delta)

        ax_err.errorbar(
            budgets,
            err_means,
            yerr=[np.array(err_means) - np.array(err_lows),
                  np.array(err_highs) - np.array(err_means)],
            fmt="o" + "-",
            capsize=4,
            #color=color,
            label=f"Optimal budget allocation"
        )

        ax_err.errorbar(
            budgets,
            err_fo_means,
            yerr=[np.array(err_fo_means) - np.array(err_fo_lows),
                  np.array(err_fo_highs) - np.array(err_fo_means)],
            fmt="s" + "--",
            capsize=4,
            #color=color,
            label=f"fine chain only"
        )

# -----------------------------
# Final styling
# -----------------------------
for ax, title in zip(
    [ax_ex, ax_wos],
    ["Optimal budget allocation", "Optimal budget allocation (WOS)"]
):
    ax.set_xscale("log")
    ax.set_xlabel("Total Budget (fine chain steps)", fontsize=13)
    ax.set_ylabel("Fraction spent on coarse level", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(ncol=2)

ax_err.set_xscale("log")
ax_err.set_xlabel("Total Budget (fine chain steps)", fontsize=13)
ax_err.set_ylabel("Averaged $W_1$ distance", fontsize=13)
ax_err.set_title(f"d = {PLOT_SINGLE_DIM}", fontsize=14)
ax_err.grid(True, which="both", linestyle="--", alpha=0.6)
ax_err.legend()

fig_ex.tight_layout()
fig_wos.tight_layout()
fig_err.tight_layout()

# -----------------------------
# Save figures and summary
# -----------------------------
summary_text = "\n".join(error_summary_lines)

for RESULTS_FOLDER in RESULTS_FOLDERS:
    fig_ex.savefig(
        os.path.join(RESULTS_FOLDER, "optimal_budget_allocation.png"),
        dpi=300, bbox_inches="tight"
    )
    fig_wos.savefig(
        os.path.join(RESULTS_FOLDER, "optimal_budget_allocation_wos.png"),
        dpi=300, bbox_inches="tight"
    )
    fig_err.savefig(
        os.path.join(
            RESULTS_FOLDER,
            f"error_vs_budget_d{PLOT_SINGLE_DIM}.png"
        ),
        dpi=300, bbox_inches="tight"
    )

    with open(os.path.join(RESULTS_FOLDER, "error_summary.txt"), "w") as f:
        f.write(summary_text)

plt.show()
