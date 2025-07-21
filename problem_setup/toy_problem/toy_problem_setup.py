from sys import exit
from problem_setup.toy_problem.problem_parameters import FwProblem

import numpy as np
from numpy import exp, log, square, sqrt, zeros
from numpy.random import standard_normal
from scipy.stats import multivariate_normal
from scipy.integrate import solve_ivp

from yagremcmc.parameter.vector import ParameterVector
from yagremcmc.statistics.interface import DensityInterface
from yagremcmc.statistics.data import Data
from yagremcmc.model.interface import SolverInterface
from yagremcmc.model.forwardModel import EvaluationStatus


class GaussianTargetDensity1d(DensityInterface):

    def __init__(self, mean, var):

        self.mean_ = mean
        self.var_ = var

    def evaluate_log(self, parameter):

        return -0.5 * square(self.mean_.coefficient -
                             parameter.coefficient) / self.var_

    def evaluate_on_mesh(self, mesh):

        return exp(-0.5 * square((self.mean_.coefficient - mesh) / self.var_))


class GaussianTargetDensity2d(DensityInterface):

    def __init__(self, mean, cov):

        self.dist_ = multivariate_normal(mean.coefficient, cov)

    def evaluate_log(self, parameter):

        return self.dist_.logpdf(parameter.coefficient)

    def evaluate_on_mesh(self, mesh):

        return self.dist_.pdf(mesh)


class toy_problem_parameter(ParameterVector):

    @classmethod
    def from_coefficient(cls, coefficient):
        return cls(coefficient)

    @classmethod
    def from_interpolation(cls, value):
        return cls(value)

    def evaluate(self):
        return self.coefficient_


class toy_problem_solver(SolverInterface):

    def __init__(self, design, config: FwProblem):

        self.design_ = design
        self.exact_fw_operator_ = config.exact_fw_operator
        self.dataShape_ = (config.n_data, config.get_data_dim())
        self._solverNoiseVar = config.solver_noise_var
        self._fw_solve_repeat_M = config.fw_solve_repeat_M

        self.param_ = None
        self.evaluation_ = None
        self.status_ = EvaluationStatus.NONE

    @property
    def status(self):
        return self.status_

    @property
    def dataShape(self):
        return self.dataShape_

    @property
    def nData(self):
        return self.dataShape_[0]

    @property
    def dataDim(self):
        return self.dataShape_[1]

    @property
    def evaluation(self):
        return self.evaluation_

    def interpolate(self, parameter):

        paramEval = parameter.evaluate()

        #self.param_ = paramEval*np.ones(self.exact_fw_operator_.shape[0])
        self.param_ = paramEval

        return

    def invoke(self):

        self.status_ = EvaluationStatus.SUCCESS

        evaluation = zeros(self.dataShape_)
        
        for i in range(self._fw_solve_repeat_M):
            for n in range(self.dataShape_[0]):
                solver_noise = sqrt(self._solverNoiseVar)*standard_normal(self.dataShape_[1])
                operator_with_design = self.design_[n,:] * self.exact_fw_operator_
                evaluation[n, :] += np.dot(operator_with_design, self.param_) + solver_noise

        self.evaluation_ = evaluation / self._fw_solve_repeat_M


def generate_synthetic_data(parameter, solver, noiseVar):

    sig = sqrt(noiseVar)

    solver.interpolate(parameter)
    solver.invoke()

    measurement = solver.evaluation + sig * standard_normal(solver.dataShape)

    return Data(measurement)
