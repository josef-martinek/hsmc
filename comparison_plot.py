import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Plot styling
fontsize = 16
markersize = 10
linewidth = 2

# === CONFIGURE PATH ===
results_folder_2s = "data/run_20251119-135029_1"
results_folder_st = "data/run_20251119-141137_r2"
results_path_2s = os.path.join(results_folder_2s, "smc_results.npz")
results_path_st = os.path.join(results_folder_st, "smc_results.npz")
data_2s = np.load(results_path_2s)
data_st = np.load(results_path_st)
json_path = os.path.join(results_folder_2s, "chain_setup.json")
target_coarse_chain_len = 8

param_seq = data_st[f"{data_st["param_seq_name"]}"]

# === LOAD CHAIN SETUPS ===
with open(json_path, "r") as f:
    chain_setup = json.load(f)


# === LOAD DATA ===
n_runs_2s = int(data_2s["n_runs"])
n_runs_fnsamp_atemp = int(data_st["n_runs"])
n_runs_st_ansamp = n_runs_fnsamp_atemp

# Runtime (all 4 methods)
runtime_ansamp_2s = data_2s["avg_runtimes_ansamp"]
runtime_fnsamp_atemp = data_st["avg_runtimes_fnsamp_atemp"]
runtime_st_ansamp = data_st["avg_runtimes_ansamp"]

# Simulation cost (3 methods)
cost_ansamp_2s = data_2s["avg_sim_cost_ansamp"]
cost_fnsamp_atemp = data_st["avg_sim_cost_fnsamp_atemp"]
cost_st_ansamp = data_st["avg_sim_cost_ansamp"]

# MSE and stddev
mse_ansamp_2s = data_2s["mse_values_ansamp"]
mse_fnsamp_atemp = data_st["mse_values_fnsamp_atemp"]
mse_st_ansamp = data_st["mse_values_ansamp"]

std_ansamp_2s = data_2s["mse_stddev_ansamp"]
std_fnsamp_atemp = data_st["mse_stddev_fnsamp_atemp"]
std_st_ansamp = data_st["mse_stddev_ansamp"]

# Compute 95% confidence intervals
def confidence_interval(mean, std, n):
    delta = 1.96 * std / np.sqrt(n)
    return mean - delta, mean + delta

ci_ansamp_2s = confidence_interval(mse_ansamp_2s, std_ansamp_2s, n_runs_2s)
ci_fnsamp_atemp = confidence_interval(mse_fnsamp_atemp,
                                      std_fnsamp_atemp,
                                      n_runs_fnsamp_atemp)
ci_st_ansamp = confidence_interval(mse_st_ansamp,
                                   std_st_ansamp,
                                   n_runs_st_ansamp)

# === PLOTS ===

# 3. MSE with CI

plt.figure(figsize=(6, 5))

# Define colors for each group
color_st_ansamp = "tab:blue"
color_c8 = "tab:red"
color_fnsamp_atemp = "tab:green"

# Extract coarse chain lengths
coarse_lengths = np.array([entry["coarse_chain_len"] for entry in chain_setup])
fine_lengths = np.array([entry["fine_chain_len"] for entry in chain_setup])

# Indices for each group
ids_c1 = np.where(coarse_lengths == 1)[0]
ids_c8 = np.where(coarse_lengths == target_coarse_chain_len)[0]

# --- Plot coarse_chain_len = 1 ---
for i, (cost, mse, ci_low, ci_high) in enumerate(
    zip(cost_st_ansamp, 
        mse_st_ansamp, 
        ci_st_ansamp[0], 
        ci_st_ansamp[1])
):
    if i == len(cost_fnsamp_atemp)-1:
        continue
    cost_vanilla = cost
    plt.plot([cost_vanilla, cost_vanilla], [ci_low, ci_high], color=color_st_ansamp, linewidth=2)
    plt.scatter(cost_vanilla, mse, color=color_st_ansamp, s=60, marker="D")  # diamond marker
    plt.text(cost_vanilla * 1.1, mse * 1.05, f"F={param_seq[i]}", fontsize=12, color=color_st_ansamp, ha="center")

# --- Plot coarse_chain_len = 8 ---
for i in ids_c8:
    # if i==ids_c8[-1]:
    #     continue
    cost = cost_ansamp_2s[i]
    mse = mse_ansamp_2s[i]
    ci_low, ci_high = ci_ansamp_2s[0][i], ci_ansamp_2s[1][i]

    plt.plot([cost, cost], [ci_low, ci_high], color=color_c8, linewidth=2)
    plt.scatter(cost, mse, marker="s", color=color_c8, s=50)
    plt.text(cost * 0.85, mse * 0.9, f"F={fine_lengths[i]}", fontsize=12, color=color_c8, ha="center")


for i, (cost, mse, ci_low, ci_high) in enumerate(
    zip(cost_fnsamp_atemp, 
        mse_fnsamp_atemp, 
        ci_fnsamp_atemp[0], 
        ci_fnsamp_atemp[1])
):
    if i > len(cost_fnsamp_atemp)-3:
        continue
    cost_vanilla = cost
    plt.plot([cost_vanilla, cost_vanilla], [ci_low, ci_high], color=color_fnsamp_atemp, linewidth=2)
    plt.scatter(cost_vanilla, mse, color=color_fnsamp_atemp, s=60, marker="D")  # diamond marker
    plt.text(cost_vanilla * 1.1, mse * 1.05, f"F={param_seq[i]}", fontsize=12, color=color_fnsamp_atemp, ha="center")

# === Axes, styling ===
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Cost", fontsize=12)
plt.ylabel("MSE on the ground truth", fontsize=12)
#plt.title("MSE with 95% CI vs cost (grouped by coarse chain length)", fontsize=13)
plt.grid(True, which="both", linestyle="--", linewidth=0.7)

# Legend
plt.scatter([], [], color=color_fnsamp_atemp, marker="D", label="no adapt. in #samp. (ESS=0.7)")
plt.scatter([], [], color=color_st_ansamp, label="adapt. #samp. (ESS=0.7)")
plt.scatter([], [], color=color_c8, marker="s", label="dtto (ESS=0.2) + c. chain (len=8)")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_folder_2s, "plot_mse_ci_comparison_ess_07.png"))

plt.show()
