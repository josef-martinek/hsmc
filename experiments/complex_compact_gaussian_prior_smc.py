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
