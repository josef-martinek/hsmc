import numpy as np
from scipy.stats import chi
from problem_setup.likelihood_base import IPModel1D, IPModelHD

from problem_setup.compact_problem.problem_parameters import SamplerConfig

class CompactModel:
    # Default dimensions & radius from problem_parameters
    m = SamplerConfig.m
    d = SamplerConfig.param_dim
    truncation_radius_R = SamplerConfig.truncation_radius_R
    base_variance = SamplerConfig.base_variance

    # Statically initialize the fixed random operator A with seed=42
    _rng = np.random.default_rng(seed=42)
    exact_fw_operator = _rng.normal(size=(m, d))

    @classmethod
    def setup(cls, m, d, truncation_radius_R, base_variance, seed=42):
        cls.m = m
        cls.d = d
        cls.truncation_radius_R = truncation_radius_R
        cls.base_variance = base_variance
        _rng = np.random.default_rng(seed=seed)
        cls.exact_fw_operator = _rng.normal(size=(m, d))

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
        Noise e is drawn from N(0, (1/nsamp) * Sigma) truncated to ||e||_2 <= R.
        Uses Inverse Transform Sampling via chi distribution for fully vectorized, non-rejection sampling.
        params shape: (N_batch, d)
        Returns shape: (N_batch, m)
        """
        u_exact = cls.exact_sol(params)
        
        # Calculate scaled isotropic variance and standard deviation (parameter-independent scalar)
        variances = cls.base_variance / nsamp
        stds = np.sqrt(variances)
        
        N_batch = params.shape[0]
        m = cls.exact_fw_operator.shape[0]
        
        # 1. Sample direction uniformly on the unit sphere
        z = np.random.randn(N_batch, m)
        directions = z / np.linalg.norm(z, axis=1, keepdims=True)
        
        # 2. Sample magnitude using Inverse Transform Sampling on a truncated Chi distribution
        # Scale the truncation radius to standard normal space
        R_scaled = cls.truncation_radius_R / stds
        
        # Get the CDF value at the truncation boundary
        upper_cdf = chi.cdf(R_scaled, df=m)
        
        # Draw uniform samples between 0 and the upper CDF bound
        u = np.random.uniform(0, upper_cdf, size=N_batch)
        
        # Use PPF (inverse CDF) to get standard magnitudes
        mag_standard = chi.ppf(u, df=m)
        
        # Scale magnitudes back to target variance
        magnitudes = mag_standard * stds
        
        # Combine magnitudes and directions
        e = directions * magnitudes[:, None]
        
        return u_exact + e


class IPModelExact(IPModel1D):
    def _evaluate_model(self, param):
        # param shape: (N,) representing 1D parameter values
        param_2d = param.reshape(-1, 1)
        return CompactModel.exact_sol(param_2d).T

    @property
    def parallel_eval(self):
        return True

    @property
    def nsamp(self):
        raise AttributeError("nsamp not used in exact model.")

    @nsamp.setter
    def nsamp(self, value):
        raise AttributeError("nsamp not used in exact model.")


class IPModelInexactFix(IPModel1D):
    def __init__(self, prior, data_noise_cov, data, nsamp):
        super().__init__(prior, data_noise_cov, data)
        self._nsamp = nsamp

    def _evaluate_model(self, param):
        # param shape: (N,)
        param_2d = param.reshape(-1, 1)
        return CompactModel.inexact_sol(param_2d, self.nsamp).T

    @property
    def parallel_eval(self):
        return True

    @property
    def nsamp(self):
        return self._nsamp

    @nsamp.setter
    def nsamp(self, value):
        raise AttributeError("nsamp is read only in model with fixed number of samples.")


class IPModelInexactAdapt(IPModel1D):
    def __init__(self, prior, data_noise_cov, data):
        super().__init__(prior, data_noise_cov, data)
        self._nsamp = None

    def _evaluate_model(self, param):
        # param shape: (N,)
        param_2d = param.reshape(-1, 1)
        return CompactModel.inexact_sol(param_2d, self.nsamp).T

    @property
    def parallel_eval(self):
        return True

    @property
    def nsamp(self):
        if self._nsamp is None:
            raise AttributeError("nsamp not specified in model with adaptive number of samples")
        return self._nsamp

    @nsamp.setter
    def nsamp(self, value):
        self._nsamp = value


class IPModelExactHD(IPModelHD):
    def _evaluate_model(self, param):
        # param shape: (N, d)
        return CompactModel.exact_sol(param).T

    @property
    def parallel_eval(self):
        return True

    @property
    def nsamp(self):
        raise AttributeError("nsamp not used in exact model.")

    @nsamp.setter
    def nsamp(self, value):
        raise AttributeError("nsamp not used in exact model.")


class IPModelInexactFixHD(IPModelHD):
    def __init__(self, prior, data_noise_cov, data, nsamp):
        super().__init__(prior, data_noise_cov, data)
        self._nsamp = nsamp

    def _evaluate_model(self, param):
        # param shape: (N, d)
        return CompactModel.inexact_sol(param, self.nsamp).T

    @property
    def parallel_eval(self):
        return True

    @property
    def nsamp(self):
        return self._nsamp

    @nsamp.setter
    def nsamp(self, value):
        raise AttributeError("nsamp is read only in model with fixed number of samples.")


class IPModelInexactAdaptHD(IPModelHD):
    def __init__(self, prior, data_noise_cov, data):
        super().__init__(prior, data_noise_cov, data)
        self._nsamp = None

    def _evaluate_model(self, param):
        # param shape: (N, d)
        return CompactModel.inexact_sol(param, self.nsamp).T

    @property
    def parallel_eval(self):
        return True

    @property
    def nsamp(self):
        if self._nsamp is None:
            raise AttributeError("nsamp not specified in model with adaptive number of samples")
        return self._nsamp

    @nsamp.setter
    def nsamp(self, value):
        self._nsamp = value
