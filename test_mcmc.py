'''
Initialize both chains
Run them
Plot results
'''

from particles.smc_samplers import MCMCSequenceWF, ThetaParticles
from core.mcmc import NoisyMetropolis, RWUncalibrated
from particles.smc_samplers import ArrayRandomWalk, Tempering
from particles import distributions as dists
import numpy as np
from problem_setup.toy_problem.problem_parameters import SamplerConfig as sc
from problem_setup.toy_problem.likelihood_model import IPModelExact, IPModelInexactFix, IPModelInexactAdapt
from problem_setup.toy_problem.likelihood_model import MatVecModel as ModelExact
import matplotlib.pyplot as plt
import seaborn as sb

np.random.seed(seed=1001)
thinning_step = 5
burn_in = 100

prior = dists.StructDist({'param': dists.Normal(loc=0., 
                                                scale=sc.prior_marg_var)})
# Generate synthetic data using ground truth
true_sol = ModelExact.exact_sol(sc.ground_truth_par).reshape(-1,1)
#true_sol = ModelExact.exact_binned_sol(sc.ground_truth_par)
observed_data = true_sol + np.random.normal(loc=0., 
                                            scale=np.sqrt(sc.data_noise_var_seq[0]), 
                                            size=true_sol.shape)

model_fine = IPModelInexactFix(prior=prior,
                          data_noise_cov=sc.data_noise_var_seq[0],
                          data=observed_data,
                          nsamp=250)

x0 = ThetaParticles(theta=prior.rvs(size=1))
x0_coarse = x0.copy()
x0_fine = x0.copy()
# If size above > 1, then the code concatenates the results in some way, don't use for now

aux_temp_alg_fine = Tempering(model=model_fine, exponents=np.array([0.,1.]))
target_fine = Tempering.current_target(aux_temp_alg_fine, 1.)
target_fine(x0_fine)

model_coarse = IPModelInexactFix(prior=prior,
                          data_noise_cov=sc.data_noise_var_seq[0],
                          data=observed_data,
                          nsamp=10)
aux_temp_alg_coarse = Tempering(model=model_coarse, exponents=np.array([0.,1.]))
target_coarse = Tempering.current_target(aux_temp_alg_coarse, 1.)
target_coarse(x0_coarse)

pmmcmc = MCMCSequenceWF(mcmc=RWUncalibrated(), len_chain=10000)
mcwm = MCMCSequenceWF(mcmc=NoisyMetropolis(), len_chain=10000)

print("Running pseudo-marginal MCMC...")
samp_pm = pmmcmc(x=x0_fine, target=target_fine)
pm_acc_rate = np.mean(samp_pm.shared['acc_rates'][0])
samp_pm = samp_pm.theta['param']
samp_pm = samp_pm[burn_in::thinning_step]
print(f"Pseudo-marginal MCMC finished. Acc rate: {pm_acc_rate}")
print("Running MCWM...")
samp_mcwm = mcwm(x=x0_coarse, target=target_coarse)
mcwm_acc_rate = np.mean(samp_mcwm.shared['acc_rates'][0])
samp_mcwm = samp_mcwm.theta['param']
samp_mcwm = samp_mcwm[burn_in::thinning_step]
print(f"MCWM finished. Acc rate: {mcwm_acc_rate}")
print("Simulation finished.")

fig, ax = plt.subplots(figsize=(6, 5))
sb.histplot(x=samp_pm, stat='density', label='PMMCMC samples', ax=ax)
sb.histplot(x=samp_mcwm, stat='density', label='MCWM samples', ax=ax)
ax.legend()
plt.show()