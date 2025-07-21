import numpy as np
import scipy.stats as stats
from particles.particles import distributions as dists
from particles.particles import smc_samplers as ssp
from particles.particles.core import SMC
from problem_setup.toy_problem.problem_parameters import MRWConfig as SamplerConfig
import problem_setup.toy_problem.problem_parameters as params
import utils.logging as logg
from matplotlib import pyplot as plt
import seaborn as sb
from utils.memoisation import likelihoodCache

logger = logg.create_custom_logger("smc_logger")
np.random.seed(1111)



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


if __name__ == "__main__":
    Nmin = 40
    Nmax = 1000000

    sc = SamplerConfig()
    parameterDim = len(sc.init_state)
    # Define prior distribution
    # prior = dists.StructDist({
    #     'param': dists.MvNormal(loc=np.zeros(parameterDim), cov=np.eye(parameterDim) * sc.prior_marg_var)
    # })
    prior = dists.StructDist({'param': dists.Normal(loc=0., 
                                                    scale=sc.prior_marg_var)})
    exact_fw_op = params.FwProblemExact().exact_fw_operator

    # Generate synthetic data using ground truth
    #observed_data = exact_fw_op @ (sc.ground_truth_par * np.ones((exact_fw_op.shape[1],1))) + np.random.normal(loc=0., scale=np.sqrt(sc.data_noise_var+params.FwProblemInexact(n_particles=n_samples).solver_noise_var), size=exact_fw_op.shape)
    observed_data = exact_fw_op @ (sc.ground_truth_par * np.ones((exact_fw_op.shape[1],1))) + np.random.normal(loc=0., scale=np.sqrt(sc.data_noise_var), size=exact_fw_op.shape)

    exact_model = IPModelExact(fw_operator=exact_fw_op, prior=prior, data_noise_var=sc.data_noise_var, data=observed_data)
    inexact_model = IPModelInexact(fw_model=params.FwProblemInexactAdapt(), prior=prior, data_noise_var=sc.data_noise_var, data=observed_data)

    # Instantiate the mode

    fk_tempering = ssp.TemperingStochSolver(Nmin=Nmin, Nmax=Nmax, solver_noise_const=params.FwProblemInexactAdapt()._solver_data_noise_ratio, model=inexact_model, wastefree=False, len_chain=10, ESSrmin=0.999)
    #fk_tempering = ssp.AdaptiveTempering(model=exact_model, wastefree=False, len_chain=10, ESSrmin=0.95)
    my_temp_alg = SMC(fk=fk_tempering, N=1000, ESSrmin=1., 
                                verbose=True)
    my_temp_alg.run()

    # for i, p in enumerate(['param']):
    #     plt.subplot(1, 2, i + 1)
    #     sb.histplot(my_temp_alg.X.theta[p], stat='density')
    #     plt.xlabel(p)
    # plt.show()

    # Compute true posterior parameters
    A = exact_fw_op  # Forward operator
    Gamma_inv = np.linalg.inv(sc.data_noise_var * np.eye(A.shape[0]))  # Inverse noise covariance
    C0_inv = np.linalg.inv(np.eye(parameterDim) * sc.prior_marg_var)  # Inverse prior covariance
    C_inv = A.T @ Gamma_inv @ A + C0_inv
    C = np.linalg.inv(C_inv)  # Posterior covariance
    m = C @ (A.T @ Gamma_inv @ observed_data + C0_inv @ np.zeros(parameterDim))  # Posterior mean

    # Define prior and posterior distributions
    prior_dist = stats.norm(loc=0, scale=sc.prior_marg_var)
    posterior_dist = stats.multivariate_normal(mean=m, cov=C)

    # Plot results
    # Create a single subplot instead of (1,2)
    fig, ax = plt.subplots(figsize=(6, 5))  # Remove (1,2), just one plot

    # Plot histogram of sampled parameters
    sb.histplot(my_temp_alg.X.theta['param'], stat='density', label='SMC Particle density', ax=ax)

    # Plot prior distribution
    x_vals = np.linspace(-3 * sc.prior_marg_var, 3 * sc.prior_marg_var, 1000)
    ax.plot(x_vals, prior_dist.pdf(x_vals), label='Prior', linestyle='dashed', color='black')

    # Plot true posterior distribution
    post_x_vals = np.linspace(m - 3 * np.sqrt(C), m + 3 * np.sqrt(C), 1000)
    ax.plot(post_x_vals.flatten(), posterior_dist.pdf(post_x_vals), label='True posterior', linestyle='solid', color='red')

    Gamma_inv = np.linalg.inv((sc.data_noise_var + params.FwProblemInexact(n_particles=Nmax)._solver_noise_var) * np.eye(A.shape[0]))  # Inverse noise covariance
    C0_inv = np.linalg.inv(np.eye(parameterDim) * sc.prior_marg_var)  # Inverse prior covariance
    C_inv = A.T @ Gamma_inv @ A + C0_inv
    C = np.linalg.inv(C_inv)  # Posterior covariance
    m = C @ (A.T @ Gamma_inv @ observed_data + C0_inv @ np.zeros(parameterDim))  # Posterior mean
    posterior_dist = stats.multivariate_normal(mean=m, cov=C)

    # Plot tempered posterior distribution
    post_x_vals = np.linspace(m - 3 * np.sqrt(C), m + 3 * np.sqrt(C), 1000)
    ax.plot(post_x_vals.flatten(), posterior_dist.pdf(post_x_vals), label='Tempered posterior', linestyle='solid', color='green')

    ax.set_xlabel('param')
    ax.legend()

    plt.show()
