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
param_seq = data[f"{data["param_seq_name"]}"]
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

post_mean_ansamp = data["post_mean_val_ansamp"]
post_mean_fnsamp_ftemp = data["post_mean_val_fnsamp_ftemp"]
post_mean_fnsamp_atemp = data["post_mean_val_fnsamp_atemp"]
post_mean_exact = data["post_mean_val_exact"]

post_mean_stddev_ansamp = data["post_mean_stddev_ansamp"]
post_mean_stddev_fnsamp_ftemp = data["post_mean_stddev_fnsamp_ftemp"]
post_mean_stddev_fnsamp_atemp = data["post_mean_stddev_fnsamp_atemp"]
post_mean_stddev_exact = data["post_mean_stddev_exact"]

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
plt.plot(param_seq, runtime_ansamp, 'o-', label="Adapt S, adapt temp", linewidth=linewidth, markersize=markersize)
plt.plot(param_seq, runtime_fnsamp_ftemp, '^-', label="Fixed #samples, fixed tempering", linewidth=linewidth, markersize=markersize)
plt.plot(param_seq, runtime_fnsamp_atemp, 's-', label="Fixed S, adapt temp", linewidth=linewidth, markersize=markersize)
plt.plot(param_seq, runtime_exact, 'd-', label="Exact", linewidth=linewidth, markersize=markersize)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(f"{data["param_seq_name"]}", fontsize=fontsize)
plt.ylabel("Average Runtime (s)", fontsize=fontsize)
plt.title(f"Runtime vs {data["param_seq_name"]}")
plt.legend(fontsize=fontsize-3, loc='upper left')
plt.grid(True)
plt.tight_layout()
#plt.xlim(param_seq[0], param_seq[-1])
plt.savefig(os.path.join(results_folder, "plot_runtime.png"))

# 2. Simulation cost
plt.figure(figsize=(6, 5))
plt.plot(param_seq, cost_ansamp, 'o-', label="Ansamp")
plt.plot(param_seq, cost_fnsamp_ftemp, '^-', label="FNSamp FTemp")
plt.plot(param_seq, cost_fnsamp_atemp, 's-', label="FNSamp ATemp")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(f"Observation {data["param_seq_name"]}")
plt.ylabel("Average Simulated Cost")
plt.title(f"Simulated Cost vs {data["param_seq_name"]}")
plt.legend(fontsize=fontsize)
plt.grid(True)
plt.tight_layout()
#plt.xlim(param_seq[0], param_seq[-1])
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
s_a, se_a, (xa, ya), (m_a, b_a) = tail_loglog_slope(param_seq, cost_ansamp, k=k)
s_f, se_f, (xf, yf), (m_f, b_f) = tail_loglog_slope(param_seq, cost_fnsamp_ftemp, k=k)
s_t, se_t, (xt, yt), (m_t, b_t) = tail_loglog_slope(param_seq, cost_fnsamp_atemp, k=k)

print(f"Ansamp slope (last 4): {s_a:.3f} ± {1.96*se_a:.3f}  (95% CI)")
print(f"FNSamp FTemp slope (last 4): {s_f:.3f} ± {1.96*se_f:.3f}  (95% CI)")
print(f"FNSamp ATemp slope (last 4): {s_t:.3f} ± {1.96*se_t:.3f}  (95% CI)")

# 3. MSE with CI
plt.figure(figsize=(6, 5))
plt.plot(param_seq, mse_ansamp, 'o-', label="Adapt S, adapt. temp", linewidth=linewidth, markersize=markersize)
plt.fill_between(param_seq, ci_ansamp[0], ci_ansamp[1], alpha=0.2)

plt.plot(param_seq, mse_fnsamp_ftemp, '^-', label="Fixed #samples, fixed tempering", linewidth=linewidth, markersize=markersize)
plt.fill_between(param_seq, ci_fnsamp_ftemp[0], ci_fnsamp_ftemp[1], alpha=0.2)

plt.plot(param_seq, mse_fnsamp_atemp, 's-', label="Fixed S, adapt. temp", linewidth=linewidth, markersize=markersize)
plt.fill_between(param_seq, ci_fnsamp_atemp[0], ci_fnsamp_atemp[1], alpha=0.2)

plt.plot(param_seq, mse_exact, 'd-', label="Exact sol, adapt temp", linewidth=linewidth, markersize=markersize)
plt.fill_between(param_seq, ci_exact[0], ci_exact[1], alpha=0.2)

plt.xscale("log")
plt.yscale("log")
plt.xlabel(f"{data["param_seq_name"]}", fontsize=fontsize)
plt.ylabel("MSE on the ground truth", fontsize=fontsize)
plt.title(f"MSE with 95% CI vs {data["param_seq_name"]}")
plt.grid(True)
plt.tight_layout()
plt.legend(fontsize=fontsize-3)
#plt.xlim(param_seq[0], param_seq[-1])
plt.savefig(os.path.join(results_folder, "plot_mse_ci.png"))

# 4. Final nsamps for Ansamp
plt.figure(figsize=(6, 5))
plt.plot(param_seq, avg_last_nsamps_ansamp, 'o-', label="Ansamp")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(f"{data["param_seq_name"]} ")
plt.ylabel("Average final sample count")
plt.title("Final Nsamps for Ansamp")
plt.grid(True)
plt.tight_layout()
plt.legend(fontsize=fontsize)
#plt.xlim(param_seq[0], param_seq[-1])
plt.savefig(os.path.join(results_folder, "plot_last_nsamps.png"))

# 5. Tempering steps
plt.figure(figsize=(6, 5))
plt.plot(param_seq, tempsteps_ansamp, 'o-', label="Adaptive #samples, adaptive tempering")
plt.plot(param_seq, tempsteps_fnsamp_atemp, 's-', label="Fixed #samples, adaptive tempering")
plt.plot(param_seq, tempsteps_exact, 'd-', label="Exact sol., adaptive tempering")
plt.xscale("log")
plt.xlabel(f"{data["param_seq_name"]}")
plt.ylabel("Average Tempering Steps used by the adaptive SMC alg.")
plt.title(f"Tempering Steps vs {data["param_seq_name"]}")
plt.legend()
plt.grid(True)
#plt.xlim(param_seq[0], param_seq[-1])
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "plot_tempering_steps.png"))

# 6. Posterior mean first component

# --- Extract first component of posterior mean from each run ---
def extract_first_component(arr):
    """Convert list/array of arrays into 1D array of their first components."""
    return np.array([a[0] for a in arr])

post_mean_ansamp_first = extract_first_component(post_mean_ansamp)
post_mean_fnsamp_ftemp_first = extract_first_component(post_mean_fnsamp_ftemp)
post_mean_fnsamp_atemp_first = extract_first_component(post_mean_fnsamp_atemp)
post_mean_exact_first = extract_first_component(post_mean_exact)

post_mean_stddev_ansamp_first = extract_first_component(post_mean_stddev_ansamp)
post_mean_stddev_fnsamp_ftemp_first = extract_first_component(post_mean_stddev_fnsamp_ftemp)
post_mean_stddev_fnsamp_atemp_first = extract_first_component(post_mean_stddev_fnsamp_atemp)
post_mean_stddev_exact_first = extract_first_component(post_mean_stddev_exact)

# --- Compute CIs for each method ---
ci_ansamp = confidence_interval(post_mean_ansamp_first, post_mean_stddev_ansamp_first, n_runs)
ci_fnsamp_ftemp = confidence_interval(post_mean_fnsamp_ftemp_first, post_mean_stddev_fnsamp_ftemp_first, n_runs)
ci_fnsamp_atemp = confidence_interval(post_mean_fnsamp_atemp_first, post_mean_stddev_fnsamp_atemp_first, n_runs)
ci_exact = confidence_interval(post_mean_exact_first, post_mean_stddev_exact_first, n_runs)

# --- Plot ---
plt.figure(figsize=(6, 5))

plt.plot(param_seq, post_mean_ansamp_first, 'o-', label="Adapt S, adapt. temp", linewidth=linewidth, markersize=markersize)
plt.fill_between(param_seq, ci_ansamp[0], ci_ansamp[1], alpha=0.2)

plt.plot(param_seq, post_mean_fnsamp_ftemp_first, '^-', label="Fixed #samples, fixed tempering", linewidth=linewidth, markersize=markersize)
plt.fill_between(param_seq, ci_fnsamp_ftemp[0], ci_fnsamp_ftemp[1], alpha=0.2)

plt.plot(param_seq, post_mean_fnsamp_atemp_first, 's-', label="Fixed S, adapt. temp", linewidth=linewidth, markersize=markersize)
plt.fill_between(param_seq, ci_fnsamp_atemp[0], ci_fnsamp_atemp[1], alpha=0.2)

plt.plot(param_seq, post_mean_exact_first, 'd-', label="Exact sol, adapt temp", linewidth=linewidth, markersize=markersize)
plt.fill_between(param_seq, ci_exact[0], ci_exact[1], alpha=0.2)

plt.xscale("log")
plt.xlabel(f"{data['param_seq_name']}", fontsize=fontsize)
plt.ylabel("Posterior mean (1st component)", fontsize=fontsize)
plt.title(f"First posterior mean component with 95% CI vs {data['param_seq_name']}")
plt.grid(True)
plt.tight_layout()
plt.legend(fontsize=fontsize - 3)

plt.savefig(os.path.join(results_folder, "plot_postmean_first_ci.png"))

plt.show()
