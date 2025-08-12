import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIGURE PATH ===
results_folder = "data/run_20250728-231738"
results_path = os.path.join(results_folder, "smc_results.npz")
data = np.load(results_path)

# === LOAD DATA ===
noise_vars = data["data_noise_var_seq"]
n_runs = int(data["n_runs"])

# Runtime (all 4 methods)
runtime_ansamp = data["avg_runtimes_ansamp"]
runtime_fnsamp_ftemp = data["avg_runtimes_fnsamp_ftemp"]
runtime_fnsamp_atemp = data["avg_runtimes_fnsamp_atemp"]
runtime_exact = data["avg_runtimes_exact"]

# Simulation cost (3 methods)
cost_ansamp = data["avg_sim_cost_ansamp"]
cost_fnsamp_ftemp = data["avg_sim_cost_fnsamp_ftemp"]
cost_fnsamp_atemp = data["avg_sim_cost_fnsamp_atemp"]

# MSE and stddev
mse_ansamp = data["mse_values_ansamp"]
mse_fnsamp_ftemp = data["mse_values_fnsamp_ftemp"]
mse_fnsamp_atemp = data["mse_values_fnsamp_atemp"]
mse_exact = data["mse_values_exact"]

std_ansamp = data["mse_stddev_ansamp"]
std_fnsamp_ftemp = data["mse_stddev_fnsamp_ftemp"]
std_fnsamp_atemp = data["mse_stddev_fnsamp_atemp"]
std_exact = data["mse_stddev_exact"]

# Compute 95% confidence intervals
def confidence_interval(mean, std, n):
    delta = 1.96 * std / np.sqrt(n)
    return mean - delta, mean + delta

ci_ansamp = confidence_interval(mse_ansamp, std_ansamp, n_runs)
ci_fnsamp_ftemp = confidence_interval(mse_fnsamp_ftemp, std_fnsamp_ftemp, n_runs)
ci_fnsamp_atemp = confidence_interval(mse_fnsamp_atemp, std_fnsamp_atemp, n_runs)
ci_exact = confidence_interval(mse_exact, std_exact, n_runs)

# Other stats
avg_last_nsamps_ansamp = data["avg_last_nsamps_ansamp"]
tempsteps_ansamp = data["avg_n_temp_steps_ansamp"]
tempsteps_fnsamp_atemp = data["avg_n_temp_steps_fnsamp_atemp"]
tempsteps_exact = data["avg_n_temp_steps_exact"]

# === PLOTS ===

# 1. Runtime
plt.figure(figsize=(6, 5))
plt.plot(noise_vars, runtime_ansamp, 'o-', label="Adaptive #samples, adaptive tempering")
plt.plot(noise_vars, runtime_fnsamp_ftemp, '^-', label="Fixed #samples, fixed tempering")
plt.plot(noise_vars, runtime_fnsamp_atemp, 's-', label="Fixed #samples, adaptive tempering")
#plt.plot(noise_vars, runtime_exact, 'd-', label="Exact")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Observation noise variance $\sigma^2$")
plt.ylabel("Average Runtime (s)")
plt.title("Runtime vs Noise Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(noise_vars[0], noise_vars[-1])
plt.savefig(os.path.join(results_folder, "plot_runtime.png"))

# 2. Simulation cost
plt.figure(figsize=(6, 5))
plt.plot(noise_vars, cost_ansamp, 'o-', label="Ansamp")
plt.plot(noise_vars, cost_fnsamp_ftemp, '^-', label="FNSamp FTemp")
plt.plot(noise_vars, cost_fnsamp_atemp, 's-', label="FNSamp ATemp")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Observation noise variance $\sigma^2$")
plt.ylabel("Average Simulated Cost")
plt.title("Simulated Cost vs Noise Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(noise_vars[0], noise_vars[-1])
plt.savefig(os.path.join(results_folder, "plot_sim_cost.png"))

# 3. MSE with CI
plt.figure(figsize=(6, 5))
plt.plot(noise_vars, mse_ansamp, 'o-', label="Adaptive #samples, adaptive tempering")
plt.fill_between(noise_vars, ci_ansamp[0], ci_ansamp[1], alpha=0.2)

plt.plot(noise_vars, mse_fnsamp_ftemp, '^-', label="Fixed #samples, fixed tempering")
plt.fill_between(noise_vars, ci_fnsamp_ftemp[0], ci_fnsamp_ftemp[1], alpha=0.2)

plt.plot(noise_vars, mse_fnsamp_atemp, 's-', label="Fixed #samples, adaptive tempering")
plt.fill_between(noise_vars, ci_fnsamp_atemp[0], ci_fnsamp_atemp[1], alpha=0.2)

plt.plot(noise_vars, mse_exact, 'd-', label="analytic. HE sol, adaptive tempering")
plt.fill_between(noise_vars, ci_exact[0], ci_exact[1], alpha=0.2)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Observation noise variance")
plt.ylabel("MSE on the true parameter $\sigma^2$")
plt.title("MSE with 95% CI vs Noise Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(noise_vars[0], noise_vars[-1])
plt.savefig(os.path.join(results_folder, "plot_mse_ci.png"))

# 4. Final nsamps for Ansamp
plt.figure(figsize=(6, 5))
plt.plot(noise_vars, avg_last_nsamps_ansamp, 'o-', label="Ansamp")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Observation noise variance ")
plt.ylabel("Average final sample count")
plt.title("Final Nsamps for Ansamp")
plt.grid(True)
plt.tight_layout()
plt.xlim(noise_vars[0], noise_vars[-1])
plt.savefig(os.path.join(results_folder, "plot_last_nsamps.png"))

# 5. Tempering steps
plt.figure(figsize=(6, 5))
plt.plot(noise_vars, tempsteps_ansamp, 'o-', label="Adaptive #samples, adaptive tempering")
plt.plot(noise_vars, tempsteps_fnsamp_atemp, 's-', label="Fixed #samples, adaptive tempering")
plt.plot(noise_vars, tempsteps_exact, 'd-', label="Analytic. HE sol., adaptive tempering")
plt.xscale("log")
plt.xlabel("Observation noise variance $\sigma^2$")
plt.ylabel("Average Tempering Steps used by the adaptive SMC alg.")
plt.title("Tempering Steps vs Noise Variance")
plt.legend()
plt.grid(True)
plt.xlim(noise_vars[0], noise_vars[-1])
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "plot_tempering_steps.png"))

plt.show()
