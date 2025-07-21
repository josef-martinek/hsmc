import numpy as np
import scipy.stats as stats
from particles.particles import distributions as dists
from particles.particles import smc_samplers as ssp
from particles.particles.core import SMC
from problem_setup.toy_problem_ill_posed.problem_parameters import MRWConfig as SamplerConfig
import problem_setup.toy_problem_ill_posed.problem_parameters as params
import utils.logging as logg
from matplotlib import pyplot as plt
import seaborn as sb
from utils.memoisation import likelihoodCache
import arviz as az
import pandas as pd
import seaborn as sb

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
        param_values = theta['param']

        #param_values = param_values * np.ones((self.fw_op.shape[1], param_values.shape[0]))  # Shape (d, N)
        
        # Forward model: G(u) = A u
        model_prediction = self.fw_op @ param_values.T  # Shape (m, N)
        
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

    def loglik(self, theta, t=None):
        if t is None:
            t = self.T
        else:
            raise NotImplementedError("loglikelihood is implemented only in full data case.")
        # Assume: theta['param'].shape == (N, d)
        param_values = theta['param']  # Already shape (N, d)
        assert len(param_values.shape) > 1

        # Forward model: G(u) = A u
        # Output shape should be (m, N), i.e., m-dimensional observation for each of N particles
        model_prediction = self.fw_model.evaluate(param_values.T)  # shape (m, N), transpose if needed

        # Compute residual y - G(u)
        # self.data should be shape (m, 1) or (m,)
        residual = self.data - model_prediction  # Ensures broadcasting, shape (m, N)

        # Noise covariance is sigma^2 * I, so inverse is 1/sigma^2 * I
        # Negative log-likelihood: Phi(u) = 0.5 * || y - G(u) ||^2 / sigma^2
        Phi = 0.5 * np.sum(residual ** 2, axis=0) / self.data_noise_var  # shape (N,)

        #self._cache.add(param_values, -Phi)

        # Log-likelihood
        return -Phi

if __name__ == "__main__":    
    n_samples = 1000
    len_chain = 10

    sc = SamplerConfig()
    parameterDim = len(sc.init_state)

    # Define prior distribution
    # prior = dists.StructDist({
    #     'param': dists.MvNormal(loc=np.zeros(parameterDim), cov=np.eye(parameterDim) * sc.prior_marg_var)
    # })
    prior = dists.StructDist({'param': dists.MvNormal(loc=np.array([0., 0.]), 
                                                    cov=np.array([[5. , 0.],[0. , 5.]]))})
    exact_fw_op = params.FwProblemExact().exact_fw_operator

    # Generate synthetic data using ground truth
    #observed_data = exact_fw_op @ (sc.ground_truth_par * np.ones((exact_fw_op.shape[1],1))) + np.random.normal(loc=0., scale=np.sqrt(sc.data_noise_var+params.FwProblemInexact(n_particles=n_samples).solver_noise_var), size=exact_fw_op.shape)
    observed_data = exact_fw_op @ sc.ground_truth_par.reshape(-1,1) + np.random.normal(loc=0., scale=np.sqrt(sc.data_noise_var), size=(exact_fw_op.shape[0],1))

    exact_model = IPModelExact(fw_operator=exact_fw_op, prior=prior, data_noise_var=sc.data_noise_var, data=observed_data)
    inexact_model = IPModelInexact(fw_model=params.FwProblemInexact(n_particles=n_samples), prior=prior, data_noise_var=sc.data_noise_var, data=observed_data)

    # Instantiate the mode

    #fk_tempering = ssp.AdaptiveTempering(model=inexact_model, wastefree=False, len_chain=len_chain, ESSrmin=0.8)
    #fk_tempering = ssp.AdaptiveTempering(model=inexact_model, wastefree=False, len_chain=len_chain, ESSrmin=0.8, move=ssp.AdaptiveMCMCSequence(len_chain=len_chain, mcmc=ssp.NoisyMetropolis()))
    fk_tempering = ssp.AdaptiveTempering(model=exact_model, wastefree=False, len_chain=len_chain, ESSrmin=0.8)
    my_temp_alg = SMC(fk=fk_tempering, N=1000, ESSrmin=1., 
                                verbose=True)
    my_temp_alg.run()
    my_temp_alg.resample_move()

    # for i, p in enumerate(['param']):
    #     plt.subplot(1, 2, i + 1)
    #     sb.histplot(my_temp_alg.X.theta[p], stat='density')
    #     plt.xlabel(p)
    # plt.show()

    # Extract particles
    particles = my_temp_alg.X.theta['param']  # Shape: (N_particles, d)
    N_particles, d = particles.shape

    # Compute posterior moments
    A = exact_fw_op  # Shape: (m, d)
    Gamma_inv = np.linalg.inv(sc.data_noise_var * np.eye(A.shape[0]))
    C0_inv = np.eye(d) / sc.prior_marg_var
    C_inv = A.T @ Gamma_inv @ A + C0_inv
    C = np.linalg.inv(C_inv)
    m = C @ (A.T @ Gamma_inv @ observed_data)

    # Tempered posterior (with solver noise)
    solver_noise_var = params.FwProblemInexact(n_particles=n_samples)._solver_noise_var
    Gamma_tilde_inv = np.linalg.inv((sc.data_noise_var + solver_noise_var) * np.eye(A.shape[0]))
    C_inv_tilde = A.T @ Gamma_tilde_inv @ A + C0_inv
    C_tilde = np.linalg.inv(C_inv_tilde)
    m_tilde = C_tilde @ (A.T @ Gamma_tilde_inv @ observed_data)

    # ---- Convert particles to InferenceData ---- #
    df_particles = pd.DataFrame(particles, columns=[f"param_{i}" for i in range(d)])
    idata = az.convert_to_inference_data(
        {
            "param": particles[np.newaxis, :, :]  # Add chain dimension
        },
        coords={"param_dim": np.arange(d)},
        dims={"param": ["chain", "draw", "param_dim"]}
    )

    # ---- Plot posterior samples ---- #
    az.plot_pair(
        idata,
        var_names=["param"],
        kind="kde",
        marginals=True,
        divergences=False,
        point_estimate="mean",
        figsize=(6 + d, 6),
    )

    # ---- Optional: Overlay analytical contours if d == 2 ---- #
    # if d == 2:
    #     x, y = np.mgrid[
    #         m[0] - 3*np.sqrt(C[0,0]):m[0] + 3*np.sqrt(C[0,0]):100j,
    #         m[1] - 3*np.sqrt(C[1,1]):m[1] + 3*np.sqrt(C[1,1]):100j
    #     ]
    #     pos = np.dstack((x, y))
    #     true_post = stats.multivariate_normal(mean=m.ravel(), cov=C)
    #     temp_post = stats.multivariate_normal(mean=m_tilde.ravel(), cov=C_tilde)

    #     plt.contour(x, y, true_post.pdf(pos), levels=5, colors='red', linestyles='dashed', linewidths=1)
    #     plt.contour(x, y, temp_post.pdf(pos), levels=5, colors='green', linestyles='solid', linewidths=1)

    #plt.suptitle("Unweighted Posterior Samples and Gaussian Approximations", fontsize=14)
    plt.show()
