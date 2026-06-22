import numpy as np

class KernelRatioEvaluator:
    """
    Evaluates the log density ratio of the noise kernels when the fidelity S changes.
    The ratio is ln( k_{S_curr}(e) / k_{S_prev}(e) ).
    Since the noise covariance is state-independent and normalization constants cancel out
    during SMC weight normalization, we only compute the exponential term:
    - (S_curr - S_prev) * ||e||_2^2 / (2 * sigma^2)
    """
    def __init__(self, model_class):
        self.model_class = model_class
    
    def evaluate_log_ratio(self, e, S_prev, S_curr): 
        # e has shape (N, m) where N is number of particles, m is data dimension
        # or (m,) for a single particle
        sigma_sq = self.model_class.base_variance
        
        # Calculate ||e||_2^2 for each particle. e is (N, m)
        # Using axis=-1 handles both (N, m) and (m,) cases
        e_norm_sq = np.sum(e**2, axis=-1)
        
        # Calculate the unnormalized log ratio
        return -(S_curr - S_prev) * e_norm_sq / (2 * sigma_sq)

