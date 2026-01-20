import numpy as np
import matplotlib.pyplot as plt
from problem_setup.wos.problem_parameters import SamplerConfig as sc
from problem_setup.wos.likelihood_model import IPModelExactHD, IPModelInexactFixHD, IPModelInexactAdaptHD
from problem_setup.wos.likelihood_model import WOSHD as ModelExact

# --- Configuration ---
n_plot = 400
theta = np.linspace(0, 2*np.pi, n_plot)   # evaluation grid

# --- Ground truth boundary function ---
a_true = sc.ground_truth_par.reshape(1, -1)
h_true = ModelExact.param_to_coeff(a_true, theta)  # shape (1, n_plot, 1)
h_true = h_true.squeeze()  # -> (n_plot,)

# --- Posterior samples ---
# post_samp: shape (N, s)
#with np.load("data/post_samp_ref.npz") as data:
#with np.load("data/post_samp_ref_inex.npz") as data:
with np.load("data/post_samp.npz") as data:
    post_samp = data["post_samp"]
N, s = post_samp.shape

# evaluate posterior boundary function for each sample
theta_eval = theta[None, :, None]     # (1, n_plot, 1)

h_post = ModelExact.param_to_coeff(post_samp, theta_eval)   # (N, n_plot, 1)
h_post = h_post.squeeze(-1)                           # (N, n_plot)

h_post_mean = np.mean(h_post, axis=0)
h_post_std  = np.std(h_post, axis=0)
h_post_low  = h_post_mean - 1.96*h_post_std
h_post_high = h_post_mean + 1.96*h_post_std

# --- Prior (mean & CI) ---
prior_mean = np.zeros(s)
prior_cov  = sc.prior_marg_var * np.eye(s)

# sample prior for visualization (analytic is more work due to cosine mixing)
Nprior = 5000
prior_samples = np.random.multivariate_normal(prior_mean, prior_cov, size=Nprior)
h_prior = ModelExact.param_to_coeff(prior_samples, theta_eval).squeeze(-1)

h_prior_mean = np.mean(h_prior, axis=0)
h_prior_std  = np.std(h_prior, axis=0)
h_prior_low  = h_prior_mean - 1.96*h_prior_std
h_prior_high = h_prior_mean + 1.96*h_prior_std

# --- Plotting ---
plt.figure(figsize=(8,5))

# Ground truth
plt.plot(theta, h_true, 'k-', lw=2, label='Ground truth')

# Posterior mean + CI
plt.plot(theta, h_post_mean, 'C1-', lw=2, label='Posterior mean')
plt.fill_between(theta, h_post_low, h_post_high, color='C1', alpha=0.25,
                 label='Posterior 95% CI')

# Prior mean + CI
plt.plot(theta, h_prior_mean, 'C0--', lw=2, label='Prior mean')
plt.fill_between(theta, h_prior_low, h_prior_high, color='C0', alpha=0.20,
                 label='Prior 95% CI')

plt.xlabel(r'$\theta$')
plt.ylabel(r'$h(\theta)$')
plt.title('Boundary condition reconstruction')
plt.legend()
plt.tight_layout()
plt.show()
