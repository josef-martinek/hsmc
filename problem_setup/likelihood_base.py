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
        if nsamp is not None:
            self.nsamp = nsamp
        return self._loglik(theta=theta)
    
    @abstractmethod
    def _loglik(self, theta, nsamp=None):
        pass

    @property
    @abstractmethod
    def parallel_eval(self) -> bool:
        pass

    @property
    @abstractmethod
    def nsamp(self):
        pass

    @nsamp.setter
    @abstractmethod
    def nsamp(self, value):
        pass


class IPModel1D(IPModel, ABC):

    def _loglik(self, theta):
        param_values = self.get_param_from_theta(theta)
    
        model_prediction = self.evaluate_model(param_values)
        
        # Compute residual y - G(u)
        residual = self.data - model_prediction  # Shape (m, N)
        
        # Compute Phi(u) = 0.5 * || y - G(u) ||_Gamma^{-1}^2
        Phi = 0.5 * np.sum((residual ** 2) / self.data_noise_cov, axis=0)  # Shape (N,)

        # Log-likelihood
        return -Phi
    
    def evaluate_model(self, params: np.ndarray):
        evals = np.zeros(shape=(len(self.data), len(params[0])))
        if self.parallel_eval:
            evals = self._evaluate_model(params[0])
        else:
            for i in range(len(params[0])):
                evals[:, i] = self._evaluate_model(params[0][i])
        return evals
    
    def get_param_from_theta(self, theta):
        return theta['param'].reshape(1,-1)
    
    @abstractmethod
    def _evaluate_model(self, param):
        pass


class IPModelHD(IPModel1D, ABC):

    def get_param_from_theta(self, theta):
        return theta['param']
    
    def evaluate_model(self, params: np.ndarray):
        evals = np.zeros(shape=(len(self.data), params.shape[0]))
        if self.parallel_eval:
            evals = self._evaluate_model(params)
        else:
            for i in range(len(params[0])):
                evals[:, i] = self._evaluate_model(params[i])
        return evals