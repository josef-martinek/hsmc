import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Plot styling
fontsize = 16
markersize = 10
linewidth = 2

# === CONFIGURE PATH ===
results_folder = "data/run_20251022-181501"
results_path = os.path.join(results_folder, "smc_results.npz")
data = np.load(results_path)
json_path = os.path.join(results_folder, "chain_setup.json")
target_fine_chain_len = 16  # <--- choose which fine_chain_len to plot
target_coarse_chain_len = 1

# === LOAD DATA ===
n_runs = int(data["n_runs"])

# Runtime (all 4 methods)
runtime_ansamp = data["avg_runtimes_ansamp"]

# Simulation cost (3 methods)
cost_ansamp = data["avg_sim_cost_ansamp"]

# MSE and stddev
mse_ansamp = data["mse_values_ansamp"]

std_ansamp = data["mse_stddev_ansamp"]

post_mean_ansamp = data["post_mean_val_ansamp"]

post_mean_stddev_ansamp = data["post_mean_stddev_ansamp"]

# Compute 95% confidence intervals
def confidence_interval(mean, std, n):
    delta = 1.96 * std / np.sqrt(n)
    return mean - delta, mean + delta

ci_ansamp = confidence_interval(mse_ansamp, std_ansamp, n_runs)

# Other stats
avg_last_nsamps_ansamp = data["avg_last_nsamps_ansamp"]

# === PLOTS ===

# 3. MSE with CI

plt.figure(figsize=(6, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(cost_ansamp)))  # choose colormap

for i, (cost, mse, ci_low, ci_high, color) in enumerate(zip(cost_ansamp, mse_ansamp, ci_ansamp[0], ci_ansamp[1], colors)):
    # CI as vertical line
    plt.plot([cost, cost], [ci_low, ci_high], color=color, linewidth=2)
    # Point marker
    plt.scatter(cost, mse, color=color, s=50, label=f"{i}")
    # Label near point
    plt.text(cost * 1.1, mse * 1.05, str(i), fontsize=12, color=color, ha="center")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Cost", fontsize=12)
plt.ylabel("MSE on the ground truth", fontsize=12)
plt.title("MSE with 95% CI vs cost", fontsize=13)
plt.grid(True, which="both", linestyle="--", linewidth=0.7)
plt.tight_layout()

# Optional legend
plt.legend(title="Index", fontsize=9, loc="best")

plt.savefig(os.path.join(results_folder, "plot_mse_ci_intervals.png"))

# === LOAD CHAIN SETUPS ===
with open(json_path, "r") as f:
    chain_setup = json.load(f)

# === FILTER ENTRIES BY fine_chain_len ===
selected_ids = [
    entry["chain_id"]
    for entry in chain_setup
    if entry["fine_chain_len"] == target_fine_chain_len
]

# === PLOT SETUP ===
plt.figure(figsize=(6, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_ids)))

for color, i in zip(colors, selected_ids):
    cost = cost_ansamp[i]
    mse = mse_ansamp[i]
    ci_low, ci_high = ci_ansamp[0][i], ci_ansamp[1][i]

    # get label info
    coarse_len = chain_setup[i]["coarse_chain_len"]

    # CI as vertical line
    plt.plot([cost, cost], [ci_low, ci_high], color=color, linewidth=2)
    # Point marker
    plt.scatter(cost, mse, color=color, s=50, label=f"coarse={coarse_len}")
    # Label near point
    plt.text(cost * 1.1, mse * 1.05, str(coarse_len), fontsize=12, color=color, ha="center")

# === AXES & STYLING ===
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Cost", fontsize=12)
plt.ylabel("MSE on the ground truth", fontsize=12)
plt.title(f"MSE vs Cost (fine_chain_len={target_fine_chain_len})", fontsize=13)
plt.grid(True, which="both", linestyle="--", linewidth=0.7)
plt.tight_layout()

plt.legend(title="Coarse chain length", fontsize=9, loc="best")

# === SAVE ===
plt.savefig(os.path.join(results_folder, f"plot_mse_ci_intervals_fine{target_fine_chain_len}.png"))

# === FILTER ENTRIES BY coarse_chain_len ===
selected_ids = [
    entry["chain_id"]
    for entry in chain_setup
    if entry["coarse_chain_len"] == target_coarse_chain_len
]

# === PLOT SETUP ===
plt.figure(figsize=(6, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_ids)))

for color, i in zip(colors, selected_ids):
    cost = cost_ansamp[i]
    mse = mse_ansamp[i]
    ci_low, ci_high = ci_ansamp[0][i], ci_ansamp[1][i]

    fine_len = chain_setup[i]["fine_chain_len"]

    # CI as vertical line
    plt.plot([cost, cost], [ci_low, ci_high], color=color, linewidth=2)
    # Point marker
    plt.scatter(cost, mse, color=color, s=50, label=f"fine={fine_len}")
    # Label near point
    plt.text(cost * 1.1, mse * 1.05, str(fine_len), fontsize=12, color=color, ha="center")

# === AXES & STYLING ===
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Cost", fontsize=12)
plt.ylabel("MSE on the ground truth", fontsize=12)
plt.title(f"MSE vs Cost (coarse_chain_len={target_coarse_chain_len})", fontsize=13)
plt.grid(True, which="both", linestyle="--", linewidth=0.7)
plt.tight_layout()

plt.legend(title="Fine chain length", fontsize=9, loc="best")

# === SAVE ===
plt.savefig(os.path.join(results_folder, f"plot_mse_ci_intervals_coarse{target_coarse_chain_len}.png"))


ref_path = os.path.join(results_folder, "post_mean_ref.npz")
ref_data = np.load(ref_path)
post_mean = ref_data["post_mean"]

# === COMPUTE MSE ON POSTERIOR MEAN ===
# post_mean_ansamp[i] and post_mean_stddev_ansamp[i] are arrays of same shape as post_mean_ref

mse_post_mean = np.array([
    (np.linalg.norm(pm - post_mean)) ** 2 + np.sum(pm_std ** 2)
    for pm, pm_std in zip(post_mean_ansamp, post_mean_stddev_ansamp)
])

# === LOAD CHAIN SETUP ===
with open(json_path, "r") as f:
    chain_setup = json.load(f)


# === 1️⃣ Plot all chains (MSE vs cost) ===
plt.figure(figsize=(6, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(cost_ansamp)))

for i, (cost, mse, color) in enumerate(
    zip(cost_ansamp, mse_post_mean, colors)
):
    plt.scatter(cost, mse, color=color, s=50, label=f"{i}")
    plt.text(cost * 1.1, mse * 1.05, str(i), fontsize=12, color=color, ha="center")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Cost", fontsize=fontsize)
plt.ylabel("MSE on posterior mean", fontsize=fontsize)
plt.title("MSE vs cost (posterior mean)", fontsize=fontsize)
plt.grid(True, which="both", linestyle="--", linewidth=0.7)
plt.tight_layout()
plt.legend(title="Index", fontsize=fontsize - 5, loc="best")
plt.savefig(os.path.join(results_folder, "plot_postmean_ci_all.png"))


# === FILTER ENTRIES BY fine_chain_len ===
selected_ids = [
    entry["chain_id"]
    for entry in chain_setup
    if entry["fine_chain_len"] == target_fine_chain_len
]

# === PLOT ===
plt.figure(figsize=(6, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_ids)))

for color, i in zip(colors, selected_ids):
    cost = cost_ansamp[i]
    mse = mse_post_mean[i]
    coarse_len = chain_setup[i]["coarse_chain_len"]

    # CI vertical line
    plt.scatter(cost, mse, color=color, s=50, label=f"coarse={coarse_len}")
    plt.text(cost * 1.1, mse * 1.05, str(coarse_len), fontsize=12, color=color, ha="center")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Cost", fontsize=12)
plt.ylabel("MSE on posterior mean", fontsize=12)
plt.title(f"MSE of posterior mean (fine_chain_len={target_fine_chain_len})", fontsize=13)
plt.grid(True, which="both", linestyle="--", linewidth=0.7)
plt.tight_layout()
plt.legend(title="Coarse chain length", fontsize=9, loc="best")

plt.savefig(os.path.join(results_folder, f"plot_mse_postmean_fine{target_fine_chain_len}.png"))


# === 3️⃣ Plot only coarse_chain_len = 1 ===
selected_ids = [
    entry["chain_id"]
    for entry in chain_setup
    if entry["coarse_chain_len"] == target_coarse_chain_len
]

plt.figure(figsize=(6, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_ids)))

for color, i in zip(colors, selected_ids):
    cost = cost_ansamp[i]
    mse = mse_post_mean[i]
    fine_len = chain_setup[i]["fine_chain_len"]

    plt.scatter(cost, mse, color=color, s=50, label=f"fine={fine_len}")
    plt.text(cost * 1.1, mse * 1.05, str(fine_len), fontsize=12, color=color, ha="center")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Cost", fontsize=fontsize)
plt.ylabel("MSE on posterior mean", fontsize=fontsize)
plt.title(f"MSE vs Cost (coarse_chain_len={target_coarse_chain_len}, posterior mean)", fontsize=fontsize)
plt.grid(True, which="both", linestyle="--", linewidth=0.7)
plt.tight_layout()
plt.legend(title="Fine chain length", fontsize=fontsize - 5, loc="best")
plt.savefig(os.path.join(results_folder, f"plot_postmean_ci_coarse{target_coarse_chain_len}.png"))


plt.show()
