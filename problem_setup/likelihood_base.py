from particles import smc_samplers as ssp
from abc import ABC, abstractmethod
import numpy as np


class IPModel(ssp.StaticModel, ABC):
    def __init__(self, prior, data_noise_cov, data: np.ndarray):
        super().__init__(data=data, prior=prior)
        self.data_noise_cov = data_noise_cov

    def loglik(self, theta, nsamp=None, t=None):
        if t is None:
            t = self.T
        else:
            raise NotImplementedError("loglikelihood is implemented only in full data case.")
        return self._loglik(theta, nsamp=round(nsamp))
    
    @abstractmethod
    def _loglik(self, theta, nsamp=None):
        pass


class IPModel1D(IPModel, ABC):

    def _loglik(self, theta, nsamp=None):
        param_values = theta['param'].reshape(1,-1)
    
        model_prediction = self.evaluate_model(param_values, nsamp)
        
        # Compute residual y - G(u)
        residual = self.data - model_prediction  # Shape (m, N)
        
        # Compute Phi(u) = 0.5 * || y - G(u) ||_Gamma^{-1}^2
        Phi = 0.5 * np.sum((residual ** 2) / self.data_noise_cov, axis=0)  # Shape (N,)

        #self._cache.add(param_values, -Phi)

        # Log-likelihood
        return -Phi
    
    def evaluate_model(self, params: np.ndarray, nsamp=None):
        evals = np.zeros(shape=(len(self.data),len(params[0])))
        for i in range(len(params[0])):
            evals[:, i] = self._evaluate_model(params[0][i], nsamp)
        return evals
    
    @abstractmethod
    def _evaluate_model(self, param, nsamp=None):
        pass