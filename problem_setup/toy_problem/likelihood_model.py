import numpy as np
from particles import smc_samplers as ssp


class IPModelExact(ssp.StaticModel):
    def __init__(self, fw_operator, prior, data_noise_var, data):
        """
        solver: toy_problem_solver instance
            The forward solver used to generate observations.
        prior: dists.StructDist
            The prior distribution over parameters.
        data: array-like
            Observed data.
        """
        super().__init__(data=data, prior=prior)
        self.fw_op = fw_operator
        self.data_noise_var = data_noise_var

    def loglik(self, theta, t=None):
        if t is None:
            t = self.T
        else:
            raise NotImplementedError("loglikelihood is implemented only in full data case.")
        param_values = theta['param'].reshape(1,-1)

        #param_values = param_values * np.ones((self.fw_op.shape[1], param_values.shape[0]))  # Shape (d, N)
        
        # Forward model: G(u) = A u
        model_prediction = self.fw_op @ param_values  # Shape (m, N)
        
        # Compute residual y - G(u)
        residual = self.data - model_prediction  # Shape (m, N)
        
        # Compute Phi(u) = 0.5 * || y - G(u) ||_Gamma^{-1}^2
        #Phi = 0.5 * np.sum((residual ** 2) / (self.data_noise_var+params.FwProblemInexact(n_particles=n_samples).solver_noise_var), axis=0)  # Shape (N,)
        Phi = 0.5 * np.sum((residual ** 2) / (self.data_noise_var), axis=0)

        # Log-likelihood
        return -Phi
    
class IPModelInexact(ssp.StaticModel):
    def __init__(self, fw_model, prior, data_noise_var, data):
        """
        solver: toy_problem_solver instance
            The forward solver used to generate observations.
        prior: dists.StructDist
            The prior distribution over parameters.
        data: array-like
            Observed data.
        """
        super().__init__(data=data, prior=prior)
        self.fw_model = fw_model
        self.data_noise_var = data_noise_var
        #self._cache = likelihoodCache(cacheSize=3)

    def loglik(self, theta, Nsamp, t=None):
        if t is None:
            t = self.T
        else:
            raise NotImplementedError("loglikelihood is implemented only in full data case.")
        param_values = theta['param'].reshape(1,-1)
        
        #if self._cache.contains(param_values):
        #    return self._cache.retrieve(parameter=param_values)
        
        # Forward model: G(u) = A u
        model_prediction = self.fw_model.evaluate(param_values, n_samp=int(Nsamp))
        
        # Compute residual y - G(u)
        residual = self.data - model_prediction  # Shape (m, N)
        
        # Compute Phi(u) = 0.5 * || y - G(u) ||_Gamma^{-1}^2
        Phi = 0.5 * np.sum((residual ** 2) / self.data_noise_var, axis=0)  # Shape (N,)

        #self._cache.add(param_values, -Phi)

        # Log-likelihood
        return -Phi
    

class IPModelInexactFixed(ssp.StaticModel):
    def __init__(self, fw_model, prior, data_noise_var, data):
        """
        solver: toy_problem_solver instance
            The forward solver used to generate observations.
        prior: dists.StructDist
            The prior distribution over parameters.
        data: array-like
            Observed data.
        """
        super().__init__(data=data, prior=prior)
        self.fw_model = fw_model
        self.data_noise_var = data_noise_var
        #self._cache = likelihoodCache(cacheSize=3)

    def loglik(self, theta, t=None):
        if t is None:
            t = self.T
        else:
            raise NotImplementedError("loglikelihood is implemented only in full data case.")
        param_values = theta['param'].reshape(1,-1)
        
        #if self._cache.contains(param_values):
        #    return self._cache.retrieve(parameter=param_values)
        
        # Forward model: G(u) = A u
        model_prediction = self.fw_model.evaluate(param_values)
        
        # Compute residual y - G(u)
        residual = self.data - model_prediction  # Shape (m, N)
        
        # Compute Phi(u) = 0.5 * || y - G(u) ||_Gamma^{-1}^2
        Phi = 0.5 * np.sum((residual ** 2) / self.data_noise_var, axis=0)  # Shape (N,)

        #self._cache.add(param_values, -Phi)

        # Log-likelihood
        return -Phi