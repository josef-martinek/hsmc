import numpy as np
from particles import smc_samplers as ssp
from problem_setup.likelihood_base import IPModel1D
import logging


class MatVecModel:
    data_dim = 10
    parameter_dim = 1
    aux = np.ones((data_dim, parameter_dim))
    exact_fw_operator = aux/np.linalg.norm(aux)
    solver_data_noise_ratio = 100
    solver_noise_var_constant = 1.
    fw_solve_repeat_M = 1

    @classmethod
    def exact_sol(cls, params):
        return cls.exact_fw_operator @ params
    
    @classmethod
    def inexact_sol(cls, params, nsamp):
        return cls.exact_fw_operator @ params + np.mean(np.random.normal(loc=0., scale=np.sqrt(cls.solver_noise_var_constant), size=(nsamp, *cls.exact_fw_operator.shape)), axis=0)


class IPModelExact(IPModel1D):
    def _evaluate_model(self, param, nsamp=None, parallel_eval=True): #TODO get rid of this nsamp, parallel eval can be in init.
        if nsamp is not None:
            logging.warning("Number of samples was specified when getting exact solution.")
        if parallel_eval is False:
            logging.error("For exact solution only parallel evaluation is supported.")
        return MatVecModel.exact_sol(params=param.reshape(1, -1))


class IPModelInexactFix(IPModel1D):
    def __init__(self, prior, data_noise_cov, data, nsamp, parallel_eval=False):
        super().__init__(prior, data_noise_cov, data)
        self.nsamp = nsamp
        self.parallel_eval = parallel_eval

    def _evaluate_model(self, param, nsamp=None): # TODO probably get rid of this nsamp (is in init)
        if nsamp is not None:
            logging.warning("Adaptive number of samples was specified when getting solution with fixed number of samples.")
        if self.parallel_eval is False:
            logging.error("Only parallel evaluation supported.")
        return MatVecModel.inexact_sol(params=param.reshape(1, -1), nsamp=self.nsamp)


class IPModelInexactAdapt(IPModel1D):
    def _evaluate_model(self, param, nsamp=None):
        if nsamp is None:
            logging.error("Number of samples must be specified for each model evaluation separately when using adaptive number of samples.")
        if self.parallel_eval is False:
            logging.error("Only parallel evaluation supported.")
        return MatVecModel.inexact_sol(params=param.reshape(1, -1), nsamp=nsamp)