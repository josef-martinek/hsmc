import numpy as np
import matplotlib.pyplot as plt
import os

# Plot styling
fontsize = 16
markersize = 10
linewidth = 2

# === CONFIGURE PATH ===
results_folder = "data/run_20250905-145609"
results_path = os.path.join(results_folder, "smc_results.npz")
data = np.load(results_path)

# === LOAD DATA ===
noise_vars = data["data_noise_var_seq"][0:-1]
n_runs = int(data["n_runs"])

# Runtime (all 4 methods)
runtime_ansamp = data["avg_runtimes_ansamp"][0:-1]
runtime_fnsamp_ftemp = data["avg_runtimes_fnsamp_ftemp"][0:-1]
runtime_fnsamp_atemp = data["avg_runtimes_fnsamp_atemp"][0:-1]
runtime_exact = data["avg_runtimes_exact"][0:-1]

# Simulation cost (3 methods)
cost_ansamp = data["avg_sim_cost_ansamp"][0:-1]
cost_fnsamp_ftemp = data["avg_sim_cost_fnsamp_ftemp"][0:-1]
cost_fnsamp_atemp = data["avg_sim_cost_fnsamp_atemp"][0:-1]

# MSE and stddev
mse_ansamp = data["mse_values_ansamp"][0:-1]
mse_fnsamp_ftemp = data["mse_values_fnsamp_ftemp"][0:-1]
mse_fnsamp_atemp = data["mse_values_fnsamp_atemp"][0:-1]
mse_exact = data["mse_values_exact"][0:-1]

std_ansamp = data["mse_stddev_ansamp"][0:-1]
std_fnsamp_ftemp = data["mse_stddev_fnsamp_ftemp"][0:-1]
std_fnsamp_atemp = data["mse_stddev_fnsamp_atemp"][0:-1]
std_exact = data["mse_stddev_exact"][0:-1]

# Compute 95% confidence intervals
def confidence_interval(mean, std, n):
    delta = 1.96 * std / np.sqrt(n)
    return mean - delta, mean + delta

ci_ansamp = confidence_interval(mse_ansamp, std_ansamp, n_runs)
ci_fnsamp_ftemp = confidence_interval(mse_fnsamp_ftemp, std_fnsamp_ftemp, n_runs)
ci_fnsamp_atemp = confidence_interval(mse_fnsamp_atemp, std_fnsamp_atemp, n_runs)
ci_exact = confidence_interval(mse_exact, std_exact, n_runs)

# Other stats
avg_last_nsamps_ansamp = data["avg_last_nsamps_ansamp"][0:-1]
tempsteps_ansamp = data["avg_n_temp_steps_ansamp"][0:-1]
tempsteps_fnsamp_atemp = data["avg_n_temp_steps_fnsamp_atemp"][0:-1]
tempsteps_exact = data["avg_n_temp_steps_exact"][0:-1]

# === PLOTS ===

# 1. Runtime
plt.figure(figsize=(6, 5))
plt.plot(noise_vars, runtime_ansamp, 'o-', label="Adapt S, adapt temp", linewidth=linewidth, markersize=markersize)
#plt.plot(noise_vars, runtime_fnsamp_ftemp, '^-', label="Fixed #samples, fixed tempering", linewidth=linewidth, markersize=markersize)
plt.plot(noise_vars, runtime_fnsamp_atemp, 's-', label="Fixed S, adapt temp", linewidth=linewidth, markersize=markersize)
#plt.plot(noise_vars, runtime_exact, 'd-', label="Exact", linewidth=linewidth, markersize=markersize)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Observation noise variance $\sigma^2$", fontsize=fontsize)
plt.ylabel("Average Runtime (s)", fontsize=fontsize)
plt.title("Runtime vs Noise Variance")
plt.legend(fontsize=fontsize-3, loc='upper left')
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
plt.legend(fontsize=fontsize)
plt.grid(True)
plt.tight_layout()
plt.xlim(noise_vars[0], noise_vars[-1])
plt.savefig(os.path.join(results_folder, "plot_sim_cost.png"))

def tail_loglog_slope(x, y, k=4):
    """Return slope (±stderr) from linear fit on the last k points in log10 space.
       Also returns the tail x,y and (slope, intercept) for plotting."""
    x = np.asarray(x)
    y = np.asarray(y)
    # Keep only positive entries (required for logs)
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    k = min(k, len(x))
    x_tail = x[-k:]
    y_tail = y[-k:]

    X = np.log10(x_tail)
    Y = np.log10(y_tail)
    # Linear regression with covariance to get stderr
    (slope, intercept), cov = np.polyfit(X, Y, deg=1, cov=True)
    slope_stderr = float(np.sqrt(cov[0, 0]))
    return slope, slope_stderr, (x_tail, y_tail), (slope, intercept)

# Compute slopes for the three curves
k=6
s_a, se_a, (xa, ya), (m_a, b_a) = tail_loglog_slope(noise_vars, cost_ansamp, k=k)
s_f, se_f, (xf, yf), (m_f, b_f) = tail_loglog_slope(noise_vars, cost_fnsamp_ftemp, k=k)
s_t, se_t, (xt, yt), (m_t, b_t) = tail_loglog_slope(noise_vars, cost_fnsamp_atemp, k=k)

print(f"Ansamp slope (last 4): {s_a:.3f} ± {1.96*se_a:.3f}  (95% CI)")
print(f"FNSamp FTemp slope (last 4): {s_f:.3f} ± {1.96*se_f:.3f}  (95% CI)")
print(f"FNSamp ATemp slope (last 4): {s_t:.3f} ± {1.96*se_t:.3f}  (95% CI)")

# 3. MSE with CI
plt.figure(figsize=(6, 5))
plt.plot(noise_vars, mse_ansamp, 'o-', label="Adapt S, adapt. temp", linewidth=linewidth, markersize=markersize)
plt.fill_between(noise_vars, ci_ansamp[0], ci_ansamp[1], alpha=0.2)

plt.plot(noise_vars, mse_fnsamp_ftemp, '^-', label="Fixed #samples, fixed tempering", linewidth=linewidth, markersize=markersize)
plt.fill_between(noise_vars, ci_fnsamp_ftemp[0], ci_fnsamp_ftemp[1], alpha=0.2)

plt.plot(noise_vars, mse_fnsamp_atemp, 's-', label="Fixed S, adapt. temp", linewidth=linewidth, markersize=markersize)
plt.fill_between(noise_vars, ci_fnsamp_atemp[0], ci_fnsamp_atemp[1], alpha=0.2)

plt.plot(noise_vars, mse_exact, 'd-', label="Exact HE sol, adapt temp", linewidth=linewidth, markersize=markersize)
plt.fill_between(noise_vars, ci_exact[0], ci_exact[1], alpha=0.2)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Observation noise variance $\sigma^2$", fontsize=fontsize)
plt.ylabel("MSE on the ground truth", fontsize=fontsize)
plt.title("MSE with 95% CI vs Noise Variance")
plt.legend(fontsize=fontsize-3)
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
plt.legend(fontsize=fontsize)
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
