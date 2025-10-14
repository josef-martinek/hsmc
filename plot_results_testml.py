import numpy as np
import matplotlib.pyplot as plt
import os

# Plot styling
fontsize = 16
markersize = 10
linewidth = 2

# === CONFIGURE PATH ===
results_folder = "data/smc_results_test"
results_path = os.path.join(results_folder, "smc_results.npz")
data = np.load(results_path)

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
    plt.text(cost, mse * 1.05, str(i), fontsize=12, color=color, ha="center")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Cost", fontsize=12)
plt.ylabel("MSE on the ground truth", fontsize=12)
plt.title("MSE with 95% CI vs cost (interval plot)", fontsize=13)
plt.grid(True, which="both", linestyle="--", linewidth=0.7)
plt.tight_layout()

# Optional legend
plt.legend(title="Index", fontsize=9, loc="best")

plt.savefig(os.path.join(results_folder, "plot_mse_ci_intervals.png"))
plt.show()
