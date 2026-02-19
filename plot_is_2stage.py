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
    "data/run_20260202-013321_",
    "data/run_20260202-091611_",
    "data/run_20260202-095220_",
    "data/run_20260202-170656_",
    "data/run_20260202-181407_"
]

CI_Z = 1.96   # 95% CI
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
PLOT_SINGLE_DIM = 64

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

# Metric definitions
METRICS = [
    ("Mean", "W_ex_by_setup", "W_wos_by_setup", "init_Wmean_ex"),
    #("Max", "Wmax_ex_by_setup", "Wmax_wos_by_setup", "init_Wmax_ex"),
    ("Weighted", "Ww_ex_by_setup", "Ww_wos_by_setup", "init_Ww_ex")
]

# Dictionary to store data for the text summary: summary_data[metric][dimension][budget]
summary_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

cmap = plt.cm.viridis
colors = cmap(np.linspace(0.2, 0.9, len(RESULTS_FOLDERS)))

# Loop over each Metric Type
for m_label, ex_key, wos_key, init_key in METRICS:
    fig_ex, ax_ex = plt.subplots(figsize=(7, 5))
    fig_wos, ax_wos = plt.subplots(figsize=(7, 5))
    fig_err, ax_err = plt.subplots(figsize=(7, 5))
    fig_err_norm, ax_err_norm = plt.subplots(figsize=(7, 5))

    print(f"\n--- Processing Metric: {m_label} ---")
    
    metric_found_in_any_folder = False

    for idx, (color, RESULTS_FOLDER) in enumerate(zip(colors, RESULTS_FOLDERS)):
        RESULTS_FILE = os.path.join(RESULTS_FOLDER, "smc_results.npz")
        if not os.path.exists(RESULTS_FILE):
            continue
            
        data = np.load(RESULTS_FILE, allow_pickle=True)
        
        # --- BACKWARD COMPATIBILITY CHECK ---
        if ex_key not in data or wos_key not in data:
            print(f"Skipping {m_label} for {RESULTS_FOLDER}: Metric keys not found.")
            continue
        
        metric_found_in_any_folder = True

        # Load core data
        W_ex = data[ex_key]
        W_wos = data[wos_key]
        setups = data["setups"]
        init_err = data.get(init_key, 1.0)

        # Load problem dimension
        spec = importlib.util.spec_from_file_location("problem_parameters", os.path.join(RESULTS_FOLDER, "problem_parameters.py"))
        pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
        param_dim = pp.SamplerConfig.param_dim

        # Organize results
        res_ex = defaultdict(lambda: defaultdict(list))
        res_wos = defaultdict(lambda: defaultdict(list))
        for e_ex, e_wos, (b, p) in zip(W_ex, W_wos, setups):
            res_ex[b][p].append(e_ex); res_wos[b][p].append(e_wos)

        budgets = sorted(res_ex.keys())
        
        # --- COLLECT STATS FOR SUMMARY ---
        for b in budgets:
            # Exact stats
            all_p_ex = sorted(res_ex[b].keys())
            err_mat_ex = np.array([res_ex[b][p] for p in all_p_ex]).T # (runs, configs)
            summary_data[m_label][param_dim][b]['ex'] = {
                'best': np.mean(np.min(err_mat_ex, axis=1)),
                'fine': np.mean(res_ex[b][0.0]),
                'worst': np.mean(np.max(err_mat_ex, axis=1))
            }
            # WOS stats
            all_p_wos = sorted(res_wos[b].keys())
            err_mat_wos = np.array([res_wos[b][p] for p in all_p_wos]).T
            summary_data[m_label][param_dim][b]['wos'] = {
                'best': np.mean(np.min(err_mat_wos, axis=1)),
                'fine': np.mean(res_wos[b][0.0]),
                'worst': np.mean(np.max(err_mat_wos, axis=1))
            }

        # Original Plotting Code
        m_ex, lo_ex, hi_ex = optimal_percent_with_ci(res_ex, budgets)
        m_wos, lo_wos, hi_wos = optimal_percent_with_ci(res_wos, budgets)

        marker = MARKERS[idx % len(MARKERS)]
        ax_ex.errorbar(budgets, m_ex, yerr=[m_ex-lo_ex, hi_ex-m_ex], fmt=marker+"-", color=color, label=f"d={param_dim}")
        ax_wos.errorbar(budgets, m_wos, yerr=[m_wos-lo_wos, hi_wos-m_wos], fmt=marker+"--", color=color, alpha=0.7, label=f"d={param_dim}")

        if param_dim == PLOT_SINGLE_DIM:
            for ax, normalized in [(ax_err, False), (ax_err_norm, True)]:
                err_means, err_lows, err_highs = [], [], []
                err_fo_means, err_fo_lows, err_fo_highs = [], [], []
                scale = init_err if normalized else 1.0

                for b in budgets:
                    e_ex_arr = np.array([res_ex[b][p] for p in sorted(res_ex[b].keys())]).T / scale
                    best_e = np.min(e_ex_arr, axis=1)
                    fine_e = np.array(res_ex[b][0.0]) / scale
                    m, s = np.mean(best_e), np.std(best_e, ddof=1)
                    fm, fs = np.mean(fine_e), np.std(fine_e, ddof=1)
                    d, fd = CI_Z*s/np.sqrt(len(best_e)), CI_Z*fs/np.sqrt(len(fine_e))
                    err_means.append(m); err_lows.append(m-d); err_highs.append(m+d)
                    err_fo_means.append(fm); err_fo_lows.append(fm-fd); err_fo_highs.append(fm+fd)

                ax.errorbar(budgets, err_means, yerr=[np.array(err_means)-err_lows, np.array(err_highs)-err_means], fmt="o-", capsize=4, label="Optimal allocation")
                ax.errorbar(budgets, err_fo_means, yerr=[np.array(err_fo_means)-err_fo_lows, np.array(err_fo_highs)-err_fo_means], fmt="s--", capsize=4, label="fine chain only")

    # Only style/save if we actually processed data for this metric
    if metric_found_in_any_folder:
        for ax, title in [(ax_ex, f"Optimal allocation ({m_label})"), (ax_wos, f"Optimal allocation WOS ({m_label})")]:
            ax.set_xscale("log"); ax.set_xlabel("Total Budget"); ax.set_ylabel("Fraction coarse level")
            ax.set_title(title); ax.grid(True, linestyle="--"); ax.legend(ncol=2)

        for ax, title, ylab in [(ax_err, f"d={PLOT_SINGLE_DIM} ({m_label})", "Error"),
                                (ax_err_norm, f"d={PLOT_SINGLE_DIM} ({m_label} Normalized)", "Error / Initial Error")]:
            ax.set_xscale("log"); ax.set_xlabel("Total Budget"); ax.set_ylabel(ylab)
            ax.set_title(title); ax.grid(True, linestyle="--"); ax.legend()

        for rf in RESULTS_FOLDERS:
            # Check if current folder was actually processed for this specific metric
            if os.path.exists(os.path.join(rf, "smc_results.npz")):
                temp_data = np.load(os.path.join(rf, "smc_results.npz"))
                if ex_key in temp_data:
                    fig_ex.savefig(os.path.join(rf, f"opt_alloc_{m_label}.png"), dpi=300)
                    fig_wos.savefig(os.path.join(rf, f"opt_alloc_wos_{m_label}.png"), dpi=300)
                    fig_err.savefig(os.path.join(rf, f"error_vs_budget_{m_label}_d{PLOT_SINGLE_DIM}.png"), dpi=300)
                    fig_err_norm.savefig(os.path.join(rf, f"error_vs_budget_{m_label}_norm_d{PLOT_SINGLE_DIM}.png"), dpi=300)
    else:
        plt.close(fig_ex); plt.close(fig_wos); plt.close(fig_err); plt.close(fig_err_norm)

# -----------------------------
# GENERATE AND EXPORT SUMMARY
# -----------------------------
summary_lines = ["===== ERROR SUMMARY =====\n", "Results folders used:\n"]
for rf in RESULTS_FOLDERS:
    summary_lines.append(f"  - {rf}")

for m_label in summary_data:
    summary_lines.append(f"\n--- METRIC: {m_label} ---")
    dims = sorted(summary_data[m_label].keys(), reverse=True)
    for d in dims:
        summary_lines.append(f"\nDimension d = {d}")
        budgets = sorted(summary_data[m_label][d].keys())
        for b in budgets:
            ex = summary_data[m_label][d][b]['ex']
            wos = summary_data[m_label][d][b]['wos']
            summary_lines.append(
                f"  Budget={b:6.1f} | "
                f"Exact: best={ex['best']:.1e}, fine={ex['fine']:.1e}, worst={ex['worst']:.1e} | "
                f"WOS: best={wos['best']:.1e}, fine={wos['fine']:.1e}, worst={wos['worst']:.1e}"
            )

summary_text = "\n".join(summary_lines)

# Write to each folder
for rf in RESULTS_FOLDERS:
    if os.path.exists(rf):
        with open(os.path.join(rf, "error_summary.txt"), "w") as f:
            f.write(summary_text)

print("\nSummary export complete.")
plt.show()