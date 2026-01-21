import numpy as np
from problem_setup.wos.likelihood_model import WOSHD as ModelExact
from scipy.stats import wasserstein_distance   # 1D Wasserstein

# --- Configuration ---
# n_plot = 500
# theta = np.linspace(0, 2*np.pi, n_plot)
# theta_eval = theta[None, :, None]

# --- Load posterior samples ---
with np.load("data/post_samp_ref.npz") as data:
    sampA = data["post_samp"]

with np.load("data/post_samp.npz") as data:
    sampB = data["post_samp"]

NA, sA = sampA.shape
NB, sB = sampB.shape

assert sA == sB, "Both methods must use same parameter dimension"

# --- Reconstruct functions ---
#hA = ModelExact.param_to_coeff(sampA, theta_eval).squeeze(-1)   # (NA, n_plot)
#hB = ModelExact.param_to_coeff(sampB, theta_eval).squeeze(-1)   # (NB, n_plot)

# --- Compute Wasserstein per angle ---
W = np.zeros(sA)
for i in range(sA): 
    W[i] = wasserstein_distance(sampA[:, i], sampB[:, i])

# --- Aggregate to a scalar metric ---
# a) mean Wasserstein
W_mean = np.mean(W)

W_max = np.max(W)

print("Mean Wasserstein distance over theta    :", W_mean)
print("Maximum distance over angles            :", W_max)
