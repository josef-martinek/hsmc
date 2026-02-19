import numpy as np
from scipy.stats import wasserstein_distance
from problem_setup.wos.likelihood_model import WOSHD as ModelExact
from problem_setup.wos.problem_parameters import SamplerConfig as sc
import problem_setup.wos.likelihood_model as lm
import matplotlib.pyplot as plt

# -----------------------------
# Load posterior samples
# -----------------------------
with np.load("data/post_samp_ref.npz") as data:
    samp_ref = data["post_samp"]   # reference posterior

# with np.load("data/post_samp.npz") as data:
#     samp_approx = data["post_samp"]  # approximate posterior

Nref, s = samp_ref.shape
# Napp, s2 = samp_approx.shape
# assert s == s2, "Parameter dimensions must match"

# -----------------------------
# Prior sampling
# -----------------------------
prior_mean = np.zeros(s)
prior_cov  = sc.prior_marg_var * np.eye(s)

Nprior = 10000   # IMPORTANT: match sample size
prior_samples = np.random.multivariate_normal(
    prior_mean, prior_cov, size=Nprior
)

# -----------------------------
# Wasserstein: posterior vs posterior
# -----------------------------
# W_post = np.zeros(s)
# for i in range(s):
#     W_post[i] = wasserstein_distance(
#         samp_ref[:, i],
#         samp_approx[:, i]
#     )

# W_post_mean = np.mean(W_post)
# W_post_max  = np.max(W_post)

# -----------------------------
# Wasserstein: prior vs posterior
# -----------------------------
W_prior = np.zeros(s)
for i in range(s):
    W_prior[i] = wasserstein_distance(
        samp_ref[:, i],
        prior_samples[:, i]
    )


W_prior_avg = np.average(W_prior, weights=1/np.arange(1,len(W_prior)+1)**lm.WOSHD.q)
W_prior_mean = np.mean(W_prior)
W_prior_max  = np.max(W_prior)

# -----------------------------
# Report
# -----------------------------
print("=== Wasserstein distances (1D marginals) ===")
print()
# print("Posterior vs Posterior approximation:")
# print(f"  Mean Wasserstein : {W_post_mean:.3e}")
# print(f"  Max  Wasserstein : {W_post_max:.3e}")
print()
print("Prior vs Reference Posterior:")
print(f"  Average weighted Wasserstein : {W_prior_avg:.1e}")
print(f"  Mean Wasserstein : {W_prior_mean:.1e}")
print(f"  Max  Wasserstein : {W_prior_max:.1e}")
# print()
# print("Relative error (mean):")
# print(f"  W(post, approx) / W(prior, post) = {W_post_mean / W_prior_mean:.3f}")

plt.plot(W_prior, marker='x')
plt.show()