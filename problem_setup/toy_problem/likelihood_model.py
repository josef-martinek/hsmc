import numpy as np
from particles import smc_samplers as ssp
from problem_setup.likelihood_base import IPModel1D, IPModelHD
import logging
from problem_setup.toy_problem.problem_parameters import SamplerConfig as sc


class MatVecModel:
    data_dim = sc.data_dim
    parameter_dim = len(sc.ground_truth_par)

    #aux = np.ones((data_dim, parameter_dim))

    rng = np.random.default_rng(seed=22)
    aux = rng.random(size=(data_dim, parameter_dim))

    # aux = np.eye(N=data_dim, M=parameter_dim)
    
    exact_fw_operator = aux/np.linalg.norm(aux)
    solver_data_noise_ratio = 100
    solver_noise_var_constant = 1.
    fw_solve_repeat_M = 1

    @classmethod
    def exact_sol(cls, params):
        return cls.exact_fw_operator @ params
    
    @classmethod
    def inexact_sol(cls, params, nsamp):
        noise = np.random.normal(loc=0., scale=np.sqrt(cls.solver_noise_var_constant), size=(nsamp, cls.exact_fw_operator.shape[0], params.shape[1]))
        return cls.exact_sol(params) + np.mean(noise, axis=0)


class IPModelExact(IPModel1D):
    def _evaluate_model(self, param):
        return MatVecModel.exact_sol(params=param.reshape(1, -1))

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
        self._nsamp = round(nsamp)

    def _evaluate_model(self, param):
        return MatVecModel.inexact_sol(params=param.reshape(1, -1), nsamp=self.nsamp)

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
        return MatVecModel.inexact_sol(params=param.reshape(1, -1), nsamp=self.nsamp)

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
        self._nsamp = round(value)


class IPModelExactHD(IPModelHD):
    def _evaluate_model(self, param):
        return MatVecModel.exact_sol(params=np.transpose(param))

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
        self._nsamp = round(nsamp)

    def _evaluate_model(self, param):
        return MatVecModel.inexact_sol(params=np.transpose(param), nsamp=self.nsamp)

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
        return MatVecModel.inexact_sol(params=np.transpose(param), nsamp=self.nsamp)

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
        self._nsamp = round(value)
