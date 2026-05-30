import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_FOLDER = "data/run_20260529-112245_"
RESULTS_FILE = os.path.join(RESULTS_FOLDER, "smc_results.npz")

CI_Z = 1.96   # 95% CI
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
PLOT_FIXED_BUDGET = 10.0

def optimal_percent_with_ci(results_dict, budgets):
    mean_opt, ci_low, ci_high = [], [], []
    for budget in budgets:
        percents = sorted(results_dict[budget].keys())
        n_runs = len(results_dict[budget][percents[0]])
        errors = np.array([results_dict[budget][p] for p in percents]).T
        best_idx = np.argmin(errors, axis=1)
        best_percents = np.array(percents)[best_idx]
        mean = np.mean(best_percents)
        delta = CI_Z * np.std(best_percents, ddof=1) / np.sqrt(n_runs)
        mean_opt.append(mean); ci_low.append(mean - delta); ci_high.append(mean + delta)
    return np.array(mean_opt), np.array(ci_low), np.array(ci_high)

if not os.path.exists(RESULTS_FILE):
    raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

data = np.load(RESULTS_FILE, allow_pickle=True)

W_ex = data["W_ex_by_setup"]
W_wos = data["W_wos_by_setup"]
setups = data["setups"]  # [budget, percent, dim]
param_dims = data.get("param_dims", [])

# Organize results: res[dim][budget][percent] = list of errors
res_ex = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
res_wos = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for e_ex, e_wos, setup in zip(W_ex, W_wos, setups):
    if len(setup) >= 3:
        b, p, d = setup[:3]
    else:
        b, p = setup[:2]
        d = param_dims[0] if len(param_dims) > 0 else 128 # fallback
    
    res_ex[d][b][p].append(e_ex)
    res_wos[d][b][p].append(e_wos)

dims = sorted(res_ex.keys())
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.2, 0.9, len(dims)))

# 1. Optimal Allocation vs Budget (for all dims)
fig_ex, ax_ex = plt.subplots(figsize=(7, 5))
fig_wos, ax_wos = plt.subplots(figsize=(7, 5))

for idx, (color, dim) in enumerate(zip(colors, dims)):
    budgets = sorted(res_ex[dim].keys())
    m_ex, lo_ex, hi_ex = optimal_percent_with_ci(res_ex[dim], budgets)
    m_wos, lo_wos, hi_wos = optimal_percent_with_ci(res_wos[dim], budgets)

    marker = MARKERS[idx % len(MARKERS)]
    ax_ex.errorbar(budgets, m_ex, yerr=[m_ex-lo_ex, hi_ex-m_ex], fmt=marker+"-", color=color, label=f"d={int(dim)}")
    ax_wos.errorbar(budgets, m_wos, yerr=[m_wos-lo_wos, hi_wos-m_wos], fmt=marker+"--", color=color, alpha=0.7, label=f"d={int(dim)}")

for ax, title in [(ax_ex, "Optimal allocation (Mean)"), (ax_wos, "Optimal allocation WOS (Mean)")]:
    ax.set_xscale("log")
    ax.set_xlabel("Total Budget")
    ax.set_ylabel("Fraction coarse level")
    ax.set_title(title)
    ax.grid(True, linestyle="--")
    ax.legend(ncol=2)

fig_ex.savefig(os.path.join(RESULTS_FOLDER, "opt_alloc_Mean.png"), dpi=300)
fig_wos.savefig(os.path.join(RESULTS_FOLDER, "opt_alloc_wos_Mean.png"), dpi=300)
plt.close(fig_ex)
plt.close(fig_wos)

# 2. Error vs Budget for each dim
for dim in dims:
    fig_err, ax_err = plt.subplots(figsize=(7, 5))
    budgets = sorted(res_ex[dim].keys())
    
    err_means, err_lows, err_highs = [], [], []
    err_fo_means, err_fo_lows, err_fo_highs = [], [], []

    for b in budgets:
        # Best percentage for each run
        e_ex_arr = np.array([res_ex[dim][b][p] for p in sorted(res_ex[dim][b].keys())]).T
        best_e = np.min(e_ex_arr, axis=1)
        
        # Fine-only percentage is 0.0
        fine_e = np.array(res_ex[dim][b][0.0])
        
        m, s = np.mean(best_e), np.std(best_e, ddof=1)
        fm, fs = np.mean(fine_e), np.std(fine_e, ddof=1)
        d_err = CI_Z * s / np.sqrt(len(best_e))
        fd_err = CI_Z * fs / np.sqrt(len(fine_e))
        
        err_means.append(m); err_lows.append(m - d_err); err_highs.append(m + d_err)
        err_fo_means.append(fm); err_fo_lows.append(fm - fd_err); err_fo_highs.append(fm + fd_err)

    ax_err.errorbar(budgets, err_means, yerr=[np.array(err_means)-err_lows, np.array(err_highs)-err_means], fmt="o-", capsize=4, label="Optimal allocation")
    ax_err.errorbar(budgets, err_fo_means, yerr=[np.array(err_fo_means)-err_fo_lows, np.array(err_fo_highs)-err_fo_means], fmt="s--", capsize=4, label="fine chain only")
    
    ax_err.set_xscale("log")
    ax_err.set_xlabel("Total Budget")
    ax_err.set_ylabel("Error")
    ax_err.set_title(f"d={int(dim)} (Mean)")
    ax_err.grid(True, linestyle="--")
    ax_err.legend()
    
    fig_err.savefig(os.path.join(RESULTS_FOLDER, f"error_vs_budget_Mean_d{int(dim)}.png"), dpi=300)
    plt.close(fig_err)

# 3. Error vs Dimension (if multiple dimensions present)
if len(dims) > 0:
    fig_dim, ax_dim = plt.subplots(figsize=(7, 5))
    d_vals, e_best, e_fine = [], [], []
    
    for d in dims:
        if PLOT_FIXED_BUDGET in res_ex[d]:
            d_vals.append(d)
            # Compute best vs fine for the fixed budget
            e_ex_arr = np.array([res_ex[d][PLOT_FIXED_BUDGET][p] for p in sorted(res_ex[d][PLOT_FIXED_BUDGET].keys())]).T
            best_e = np.mean(np.min(e_ex_arr, axis=1))
            fine_e = np.mean(res_ex[d][PLOT_FIXED_BUDGET][0.0])
            
            e_best.append(best_e)
            e_fine.append(fine_e)
            
    if d_vals:
        ax_dim.plot(d_vals, e_best, "o-", label="Optimal allocation")
        ax_dim.plot(d_vals, e_fine, "s--", label="fine chain only")
        ax_dim.set_xscale("log")
        ax_dim.set_yscale("log")
        ax_dim.set_xlabel("Dimension d")
        ax_dim.set_ylabel("Error")
        ax_dim.set_title(f"Error vs Dimension (Budget={PLOT_FIXED_BUDGET}, Mean)")
        ax_dim.grid(True, linestyle="--")
        ax_dim.legend()
        fig_dim.savefig(os.path.join(RESULTS_FOLDER, f"error_vs_dim_Mean_budget{PLOT_FIXED_BUDGET}.png"), dpi=300)
    plt.close(fig_dim)

# Text summary
summary_lines = ["===== ERROR SUMMARY =====\n", f"Results folder used: {RESULTS_FOLDER}\n"]

for d in sorted(dims, reverse=True):
    summary_lines.append(f"\n--- Dimension d = {int(d)} ---")
    budgets = sorted(res_ex[d].keys())
    for b in budgets:
        # Exact stats
        err_mat_ex = np.array([res_ex[d][b][p] for p in sorted(res_ex[d][b].keys())]).T
        ex_best = np.mean(np.min(err_mat_ex, axis=1))
        ex_fine = np.mean(res_ex[d][b][0.0])
        ex_worst = np.mean(np.max(err_mat_ex, axis=1))
        
        # WOS stats
        err_mat_wos = np.array([res_wos[d][b][p] for p in sorted(res_wos[d][b].keys())]).T
        wos_best = np.mean(np.min(err_mat_wos, axis=1))
        wos_fine = np.mean(res_wos[d][b][0.0])
        wos_worst = np.mean(np.max(err_mat_wos, axis=1))
        
        summary_lines.append(
            f"  Budget={b:6.1f} | "
            f"Exact: best={ex_best:.1e}, fine={ex_fine:.1e}, worst={ex_worst:.1e} | "
            f"WOS: best={wos_best:.1e}, fine={wos_fine:.1e}, worst={wos_worst:.1e}"
        )

with open(os.path.join(RESULTS_FOLDER, "error_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))

print("Plots and summary generated successfully.")
