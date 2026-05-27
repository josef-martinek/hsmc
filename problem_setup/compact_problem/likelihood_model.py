import numpy as np
from scipy.stats import chi

class CompactModel:
    exact_fw_operator = None
    truncation_radius_R = None

    @classmethod
    def setup(cls, m, d, R, seed=42):
        """
        Set up the problem dimensions and operator.
        """
        np.random.seed(seed)
        cls.exact_fw_operator = np.random.randn(m, d)
        cls.truncation_radius_R = R

    @classmethod
    def state_variance(cls, params):
        """
        Compute the state-dependent isotropic variance: 2.0 + sin(||u||_2)
        params shape: (N_batch, d)
        Returns shape: (N_batch,)
        """
        norms = np.linalg.norm(params, axis=1)
        return 2.0 + np.sin(norms)

    @classmethod
    def exact_sol(cls, params):
        """
        Compute the exact forward map: G(u) = A * u
        params shape: (N_batch, d)
        Returns shape: (N_batch, m)
        """
        return params @ cls.exact_fw_operator.T

    @classmethod
    def inexact_sol(cls, params, nsamp):
        """
        Stochastic forward solver G_N(u, e) = G(u) + e.
        Noise e is drawn from N(0, (1/nsamp) * Sigma(u)) truncated to ||e||_2 <= R.
        Uses Inverse Transform Sampling via chi distribution for fully vectorized, non-rejection sampling.
        params shape: (N_batch, d)
        Returns shape: (N_batch, m)
        """
        u_exact = cls.exact_sol(params)
        
        # Calculate scaled isotropic variance and standard deviation for each batch item
        variances = cls.state_variance(params) / nsamp
        stds = np.sqrt(variances)  # shape (N_batch,)
        
        N_batch = params.shape[0]
        m = cls.exact_fw_operator.shape[0]
        
        # 1. Sample direction uniformly on the unit sphere
        z = np.random.randn(N_batch, m)
        directions = z / np.linalg.norm(z, axis=1, keepdims=True)
        
        # 2. Sample magnitude using Inverse Transform Sampling on a truncated Chi distribution
        # Scale the truncation radius to standard normal space (since stds differ per parameter)
        R_scaled = cls.truncation_radius_R / stds
        
        # Get the CDF value at the truncation boundary
        upper_cdf = chi.cdf(R_scaled, df=m)
        
        # Draw uniform samples between 0 and the upper CDF bound
        u = np.random.uniform(0, upper_cdf)
        
        # Use PPF (inverse CDF) to get standard magnitudes
        mag_standard = chi.ppf(u, df=m)
        
        # Scale magnitudes back to target variance
        magnitudes = mag_standard * stds
        
        # Combine magnitudes and directions
        e = directions * magnitudes[:, None]
        
        return u_exact + e
