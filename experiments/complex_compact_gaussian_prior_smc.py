import numpy as np
from particles import distributions as dists
from experiments.complex_compact_smc import ComplexCompactSMCExperiment

class ComplexCompactGaussianPriorSMCExperiment(ComplexCompactSMCExperiment):
    def get_prior(self, problem_cfg):
        mu = problem_cfg.prior_mu
        sigma = problem_cfg.prior_sigma
        if self.param_dim == 1:
            prior = dists.StructDist({'param': dists.Normal(loc=mu, scale=sigma)})
        else:
            prior = dists.StructDist({'param': dists.IndepProd(*(dists.Normal(loc=mu, scale=sigma) for _ in range(self.param_dim)))})
        return prior

    def sample_ground_truth(self, problem_cfg, rng_truth):
        return rng_truth.normal(loc=problem_cfg.prior_mu, scale=problem_cfg.prior_sigma, size=self.param_dim)

    def compute_reference_solution(self, model_ex, observed_data, cfg, problem_cfg):
        from utils.exact_gaussian_posterior import get_exact_gaussian_posterior_samples
        from problem_setup.compact_problem.likelihood_model import CompactModel
        return get_exact_gaussian_posterior_samples(
            A=CompactModel.exact_fw_operator,
            b=np.zeros(problem_cfg.m),
            y=observed_data.flatten(),
            mu_0=problem_cfg.prior_mu * np.ones(problem_cfg.param_dim),
            Sigma_0=(problem_cfg.prior_sigma ** 2) * np.eye(problem_cfg.param_dim),
            sigma_noise_sq=problem_cfg.data_noise_var,
            N=cfg.experiment.ref_sol.N,
            gamma=1.0
        )
