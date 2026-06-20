import numpy as np
from problem_setup.compact_problem.likelihood_model import CompactModel, IPModelInexactAdapt, IPModelInexactFix, IPModelExact, IPModelInexactAdaptHD, IPModelInexactFixHD

class LikelihoodNoiseTrackingMixin:
    """
    Mixin for IPModel classes to intercept the likelihood evaluation
    and explicitly save the generated noise realization `e` into the
    particle's state array (`theta['e']`).
    """
    def _loglik(self, theta, nsamp=None):
        # We need to extract parameters to evaluate the model
        param_values = self.get_param_from_theta(theta)
        
        if nsamp is not None:
            model_prediction = self.evaluate_model(param_values, nsamp)
        else:
            model_prediction = self.evaluate_model(param_values)
        
        # Inject the noise realization directly into the state array.
        # This overrides the dummy values drawn by the particles prior.
        if 'e' in theta.dtype.names:
            theta['e'] = self.get_noise_realization(param_values, model_prediction)
            
        # Standard log-likelihood computation
        residual = self.data - model_prediction
        Phi = 0.5 * np.sum((residual ** 2) / self.data_noise_cov, axis=0)
        return -Phi


class IPModelInexactAdaptDensity(LikelihoodNoiseTrackingMixin, IPModelInexactAdapt):
    """
    Adaptive inexact model that tracks the internal noise realization e.
    """
    def get_noise_realization(self, param, model_prediction):
        # param shape is (1, N) from IPModel1D. model_prediction is (m, N)
        param_2d = param.reshape(-1, 1)
        # model_prediction is (m, N). exact_sol returns (N, m).
        # We want to return e of shape (N, m) because ThetaParticles expects
        # array parameters for N particles to have shape (N, ...).
        # So e for N particles has shape (N, m).
        return model_prediction.T - CompactModel.exact_sol(param_2d)

class IPModelInexactDensity(LikelihoodNoiseTrackingMixin, IPModelInexactFix):
    """
    Fixed inexact model that tracks the internal noise realization e.
    """
    def get_noise_realization(self, param, model_prediction):
        param_2d = param.reshape(-1, 1)
        return model_prediction.T - CompactModel.exact_sol(param_2d)

class IPModelInexactAdaptHDDensity(LikelihoodNoiseTrackingMixin, IPModelInexactAdaptHD):
    """
    Adaptive inexact model HD that tracks the internal noise realization e.
    """
    def get_noise_realization(self, param, model_prediction):
        return model_prediction.T - CompactModel.exact_sol(param)

class IPModelInexactHDDensity(LikelihoodNoiseTrackingMixin, IPModelInexactFixHD):
    """
    Fixed inexact model HD that tracks the internal noise realization e.
    """
    def get_noise_realization(self, param, model_prediction):
        return model_prediction.T - CompactModel.exact_sol(param)

class IPModelExactDensity(IPModelExact):
    """
    Exact model (no noise). Passed through for compatibility.
    """
    pass
