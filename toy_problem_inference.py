import sys
sys.path.insert(0, 'C:\\Users\\jmart\\Documents\\Python\\pivo\\yagre-mcmc')

import toy_problem_setup as setup
import arviz as az

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import uniform
from yagremcmc.model.forwardModel import ForwardModel
from yagremcmc.chain.method.mrw import MRWBuilder
from yagremcmc.chain.method.pcn import PCNBuilder
from yagremcmc.chain.method.am import AMBuilder
from yagremcmc.statistics.gaussian import Gaussian
from yagremcmc.statistics.covariance import DiagonalCovarianceMatrix, IIDCovarianceMatrix
from yagremcmc.statistics.noise import CentredGaussianIIDNoise
from yagremcmc.statistics.likelihood import AdditiveNoiseLikelihood
from yagremcmc.statistics.bayesModel import BayesianRegressionModel
from yagremcmc.postprocessing.autocorrelation import integrated_autocorrelation_nd

np.random.seed(1111)

# available options are 'mrw', 'pcn'
method = 'mrw'

if method != 'pcn':

    # available options are 'iid', 'indep'. For 'am',
    # this will be used as the initial covariance.
    proposalCovType = 'iid'

# define model problem
config_inference = {
    'a': 1,
    'b': 0,
    'c': 0,
    'd': 1,
    'nData': 1,
    'dataDim': 2,
    'solver_noise': 1,
    }
config_truth = {
    'a': 1,
    'b': 0,
    'c': 0,
    'd': 1,
    'nData': 1,
    'dataDim': 2,
    'solver_noise': 0,
    }
design = np.array([uniform(0.5, 1.5, 2) for _ in range(config_inference['nData'])])

# define forward problem
solver_inference = setup.toy_problem_solver(design, config_inference)
fwdModel = ForwardModel(solver_inference)

solver_truth = setup.toy_problem_solver(design, config_truth)

# define problem parameters
parameterDim = 1
groundTruth = setup.toy_problem_parameter.from_interpolation(
    np.array([2]))
assert groundTruth.dimension == parameterDim

# generate data
dataNoiseVar = 0.8
data = setup.generate_synthetic_data(groundTruth, solver_truth, dataNoiseVar)

print("synthetic data generated")

# start with a prior centred around the true parameter coefficient
priorMean = setup.toy_problem_parameter.from_coefficient(np.zeros(1))

priorMargVar = 2
priorCovariance = IIDCovarianceMatrix(parameterDim, priorMargVar)

# set up prior
prior = Gaussian(priorMean, priorCovariance)

# define a noise model
noiseVariance = dataNoiseVar
noiseModel = CentredGaussianIIDNoise(noiseVariance)

# define the statistical inverse problem
likelihood = AdditiveNoiseLikelihood(data, fwdModel, noiseModel)
statModel = BayesianRegressionModel(prior, likelihood)

# configure the chain setup
if method == 'pcn':

    chainBuilder = PCNBuilder()
    chainBuilder.stepSize = 0.01

elif method == 'mrw':

    if proposalCovType == 'iid':

        proposalMargVar = 0.15
        proposalCov = IIDCovarianceMatrix(parameterDim, proposalMargVar)

    elif proposalCovType == 'indep':

        proposalMargVar = np.array([0.02, 0.06])
        proposalCov = DiagonalCovarianceMatrix(proposalMargVar)

    else:
        raise ValueError(
            "Unknown Proposal covariance type: " + proposalCovType)

    chainBuilder = MRWBuilder()
    chainBuilder.proposalCovariance = proposalCov

else:
    raise ValueError("Unknown MCMC method: " + method)

chainBuilder.bayesModel = statModel

sampler = chainBuilder.build_method()

# run mcmc
nSteps = 50000
initState = setup.toy_problem_parameter.from_coefficient(np.array([3]))
sampler.run(nSteps, initState)

states = sampler.chain.trajectory

burnIn = 5000
thinningStep = integrated_autocorrelation_nd(states[burnIn:], 'max')

mcmcSamples = states[burnIn::thinningStep]
meanState = setup.toy_problem_parameter.from_coefficient(
    np.mean(states, axis=0))
posteriorMean = setup.toy_problem_parameter.from_coefficient(
    np.mean(mcmcSamples, axis=0))

# estimates mean
print(f"true parameter: {groundTruth.evaluate()}")
print(f"raw posterior mean: {meanState.evaluate()}")
print(f"processed posterior mean: {posteriorMean.evaluate()}")
print(f"Acceptance rate: {sampler.chain.diagnostics.global_acceptance_rate()}")
print(f"IAT estimate: {thinningStep}")
print(f"effective sample size: {(nSteps - burnIn) // thinningStep}")

# Convert mcmcSamples (list of arrays) into a 2D NumPy array
# Each row will represent a sample, and each column corresponds to a parameter
mcmcSamples_array = np.array([sample for sample in mcmcSamples])

# Convert to ArviZ InferenceData
inference_data = az.from_dict(posterior={"parameter": mcmcSamples_array.T})

# Summarize the posterior
print(az.summary(inference_data, round_to=2))

# Plot the posterior density
az.plot_posterior(inference_data)
plt.show()