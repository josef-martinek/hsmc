from sys import exit

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

    def __init__(self, design, config):

        self.initCond_ = design
        self.fixedParam_ = [config['a'], config['b'], config['c'], config['d']]
        self.dataShape_ = (config['nData'], config['dataDim'])
        self._solverNoiseVar = config['solver_noise']

        self.param_ = [None]
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

    def matrix__(self, a, b, c, d):

        return np.array([[a,b],[c,d]])

    def interpolate(self, parameter):

        paramEval = parameter.evaluate()

        self.param_ = [paramEval[0]]

        return

    def invoke(self):

        self.status_ = EvaluationStatus.SUCCESS

        a = self.fixedParam_[0]
        b = self.fixedParam_[1]
        c = self.fixedParam_[2]
        d = self.fixedParam_[3]
        coeff = self.param_[0]

        evaluation = zeros(self.dataShape_)

        forward_operator = coeff * self.matrix__(a,b,c,d)
        
        for n in range(self.dataShape_[0]):
            #evaluation[n, :] = np.dot(forward_operator,self.initCond_[n,:]) + sqrt(self._solverNoiseVar)*standard_normal(self.dataShape_[1])
            evaluation[n, :] = np.dot(forward_operator,self.initCond_[n,:]) + coeff * sqrt(self._solverNoiseVar)*standard_normal(self.dataShape_[1])

        self.evaluation_ = evaluation


def generate_synthetic_data(parameter, solver, noiseVar):

    sig = sqrt(noiseVar)

    solver.interpolate(parameter)
    solver.invoke()

    measurement = solver.evaluation + sig * standard_normal(solver.dataShape)

    return Data(measurement)
