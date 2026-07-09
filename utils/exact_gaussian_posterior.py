import numpy as np

def get_exact_gaussian_posterior_samples(A, b, y, mu_0, Sigma_0, sigma_noise_sq, N, gamma=1.0):
    """
    Computes exact posterior samples for Bayesian Linear Regression:
    Likelihood: y | u ~ N(Au + b, (sigma_noise_sq / gamma) * I)
    Prior: u ~ N(mu_0, Sigma_0)
    
    Parameters:
    A: Forward operator matrix, shape (m, d)
    b: Constant vector offset, shape (m,)
    y: Observed data, shape (m,)
    mu_0: Prior mean, shape (d,)
    Sigma_0: Prior covariance matrix, shape (d, d)
    sigma_noise_sq: Base noise variance scalar
    N: Number of exact samples to draw
    gamma: Tempered likelihood exponent (default 1.0)
    
    Returns:
    samples: Array of shape (N, d) containing exact samples from the posterior.
    """
    m, d = A.shape
    
    Sigma_e_inv = (gamma / sigma_noise_sq) * np.eye(m)
    Sigma_0_inv = np.linalg.inv(Sigma_0)
    
    # Sigma_post = (Sigma_0^-1 + A^T Sigma_e^-1 A)^-1
    Sigma_post = np.linalg.inv(Sigma_0_inv + A.T @ Sigma_e_inv @ A)
    
    # mu_post = Sigma_post (Sigma_0^-1 mu_0 + A^T Sigma_e^-1 (y - b))
    mu_post = Sigma_post @ (Sigma_0_inv @ mu_0 + A.T @ Sigma_e_inv @ (y - b))
    
    # Draw N samples
    samples = np.random.multivariate_normal(mean=mu_post, cov=Sigma_post, size=N)
    return samples
