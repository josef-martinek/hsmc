import numpy as np
import scipy.stats as stats
import seaborn as sb
import matplotlib.pyplot as plt
from juliacall import Main as jl


data = np.load("dsmc_infer_results.npz")

print("True Kn: 0.1\n")
post_mean = 10**np.mean(data["post_samp"])
print(f"Exp of posterior mean: {post_mean}")
print("Computing posterior mean solution...")
jl.seval("include(\"problem_setup/rayleigh_problem_dsmc/rayleigh.jl\")")
ps, vs, params = jl.simulate(Kn=post_mean, Nsim=100, Nmft=10, Ncell=32)
mac = jl.stats(ps, vs, params)

# Extract arrays
z_pm = np.array(mac.z)
uy_pm = np.array(mac.uy)
print("Posterior mean sol computed")

# Define prior and posterior distributions
prior_dist = stats.norm(loc=data["prior_mean"], scale=data["prior_marg_var"])

# Plot results
# Create a single subplot instead of (1,2)
plt.figure(figsize=(6,5))  # Remove (1,2), just one plot

# Plot histogram of sampled parameters
sb.histplot(x=data["post_samp"], stat='density', label='SMC Samples', ax=plt.gca())

# Plot prior distribution
x_vals = np.linspace(-3 * data["prior_marg_var"], 3 * data["prior_marg_var"], 1000)
plt.plot(x_vals, prior_dist.pdf(x_vals), label='Prior', linestyle='dashed', color='black')
plt.xlabel('log(Kn)')
plt.legend()

plt.figure(figsize=(6,5))
x_axis = z_pm
plt.plot(x_axis, data["true_sol"], label="true sol")
plt.plot(x_axis, data["noisy_data"], label="noisy data")
plt.plot(x_axis, uy_pm, label="posterior mean solution")
plt.legend()

plt.show()