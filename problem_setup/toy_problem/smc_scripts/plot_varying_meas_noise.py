import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIGURE ===
# List your result folders in the order that matches the param_values below
result_folders = [
    "data/smc_results_isstep58",
    "data/smc_results_isstep59",
    "data/smc_results_isstep60",
    "data/smc_results_isstep61",
]

# Define the parameter values used in those experiments (same order)
param_values = [10, 50, 250, 1250]  # <-- replace with your actual values

# === LOAD DATA FROM EACH FOLDER ===
avg_runtimes_adapt = []
avg_runtimes_fixed = []
mse_values_adapt = []
mse_values_fixed = []
mse_lower_adapt = []
mse_upper_adapt = []
mse_lower_fixed = []
mse_upper_fixed = []

for folder in result_folders:
    results_path = os.path.join(folder, "smc_results.npz")
    data = np.load(results_path)

    avg_runtimes_adapt.append(data["avg_runtimes_adapt"].item())  # scalar
    avg_runtimes_fixed.append(data["avg_runtimes_fixed"].item())

    mse_values_adapt.append(data["mse_values_adapt"].item())
    mse_values_fixed.append(data["mse_values_fixed"].item())

    mse_lower_adapt.append(data["mse_lower"].item())
    mse_upper_adapt.append(data["mse_upper"].item())

    mse_lower_fixed.append(data["mse_lower_fixed"].item())
    mse_upper_fixed.append(data["mse_upper_fixed"].item())

# Convert to numpy arrays
param_values = np.array(param_values)
avg_runtimes_adapt = np.array(avg_runtimes_adapt)
avg_runtimes_fixed = np.array(avg_runtimes_fixed)
mse_values_adapt = np.array(mse_values_adapt)
mse_values_fixed = np.array(mse_values_fixed)
mse_lower_adapt = np.array(mse_lower_adapt)
mse_upper_adapt = np.array(mse_upper_adapt)
mse_lower_fixed = np.array(mse_lower_fixed)
mse_upper_fixed = np.array(mse_upper_fixed)

print(f"Cost ratios for parameters (fixed/adapt) {param_values}: {avg_runtimes_fixed/avg_runtimes_adapt}")
print(f"MSE ratios (adapt/fixed): {mse_values_adapt/mse_values_fixed}")

# === PLOT ===
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# --- Average runtime ---
ax[0].plot(param_values, avg_runtimes_adapt, marker='o', linestyle='-', label='Adapt nsamp', color='blue')
ax[0].plot(param_values, avg_runtimes_fixed, marker='^', linestyle='-', label='Fixed nsamp', color='red')
#ax[0].set_xlabel("Observation error marg. var.")
ax[0].set_xlabel("Number of SMC particles")
ax[0].set_ylabel("Average Runtime (s)")
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_title("Average Runtime")
ax[0].legend()
ax[0].set_xlim(param_values[0], param_values[-1])

# --- MSE with confidence intervals ---
ax[1].plot(param_values, mse_values_adapt, marker='s', linestyle='-', color='blue', label='MSE Adapt nsamp')
ax[1].fill_between(param_values, mse_lower_adapt, mse_upper_adapt, color='blue', alpha=0.3)

ax[1].plot(param_values, mse_values_fixed, marker='^', linestyle='-', color='red', label='MSE Fixed nsamp')
ax[1].fill_between(param_values, mse_lower_fixed, mse_upper_fixed, color='red', alpha=0.3)

#ax[1].set_xlabel("Observation error marg. var.")
ax[1].set_xlabel("Number of SMC particles")
ax[1].set_ylabel("MSE")
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_title("MSE")
ax[1].legend()
ax[1].set_xlim(param_values[0], param_values[-1])

plt.tight_layout()
plt.savefig("comparison_vs_param.png")
plt.show()
