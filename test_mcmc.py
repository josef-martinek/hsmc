from particles.smc_samplers import MCMCSequenceWF, ThetaParticles
from core.mcmc import NoisyMetropolis, RWUncalibrated
from particles.smc_samplers import ArrayRandomWalk, Tempering
from particles import distributions as dists
import numpy as np
from problem_setup.toy_problem.problem_parameters import SamplerConfig as sc
from problem_setup.toy_problem.likelihood_model import IPModelInexactFix, IPModelExact
from problem_setup.toy_problem.likelihood_model import IPModelInexactFixHD, IPModelExactHD
from problem_setup.toy_problem.likelihood_model import MatVecModel as ModelExact
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

np.random.seed(seed=1111)
thinning_step = 10
burn_in = 1000
len_chain = 10000
nsamp_pm = 4000
nsamp_mcwm = 100

if len(sc.ground_truth_par) == 1:
    prior = dists.StructDist({'param': dists.Normal(loc=0., 
                                                    scale=sc.prior_marg_var)})
else:
    param_dim = len(sc.ground_truth_par)
    prior = dists.StructDist({'param': dists.MvNormal(loc=np.zeros(param_dim),
                                                    cov=sc.prior_marg_var*np.eye(param_dim))})

# Generate synthetic data using ground truth
true_sol = ModelExact.exact_sol(sc.ground_truth_par).reshape(-1,1)
observed_data = true_sol + np.random.normal(loc=0.,
                                            scale=np.sqrt(sc.data_noise_var_seq[0]), 
                                            size=true_sol.shape)

if len(sc.ground_truth_par) == 1:
    model_fine = IPModelInexactFix(prior=prior,
                            data_noise_cov=sc.data_noise_var_seq[0],
                            data=observed_data,
                            nsamp=nsamp_pm)
    model_coarse = IPModelInexactFix(prior=prior,
                            data_noise_cov=sc.data_noise_var_seq[0],
                            data=observed_data,
                            nsamp=nsamp_mcwm)
    model_exact = IPModelExact(prior=prior,
                            data_noise_cov=sc.data_noise_var_seq[0],
                            data=observed_data)
else:
    model_fine = IPModelInexactFixHD(prior=prior,
                            data_noise_cov=sc.data_noise_var_seq[0],
                            data=observed_data,
                            nsamp=nsamp_pm)
    model_coarse = IPModelInexactFixHD(prior=prior,
                            data_noise_cov=sc.data_noise_var_seq[0],
                            data=observed_data,
                            nsamp=nsamp_mcwm)
    model_exact = IPModelExactHD(prior=prior,
                            data_noise_cov=sc.data_noise_var_seq[0],
                            data=observed_data)

x0 = ThetaParticles(theta=prior.rvs(size=1))
x0_coarse = x0.copy()
x0_fine = x0.copy()
x0_exact = x0.copy()
# If size > 1 (in ThetaParticles(theta=prior.rvs(size=1))), then the code
# concatenates the results in some way, don't use for now

aux_temp_alg_fine = Tempering(model=model_fine, exponents=np.array([0., 1.]))
target_fine = Tempering.current_target(aux_temp_alg_fine, 1.)
target_fine(x0_fine)

aux_temp_alg_coarse = Tempering(model=model_coarse, exponents=np.array([0., 1.]))
# Some exponents have to be here, but not relevant which ones
target_coarse = Tempering.current_target(aux_temp_alg_coarse, 1.)
target_coarse(x0_coarse) #Initialise likelihood for the starting point

aux_temp_alg_exact = Tempering(model=model_exact, exponents=np.array([0., 1.]))
target_exact = Tempering.current_target(aux_temp_alg_exact, 1.)
target_exact(x0_exact)

pmmcmc = MCMCSequenceWF(mcmc=RWUncalibrated(), len_chain=len_chain)
mcwm = MCMCSequenceWF(mcmc=NoisyMetropolis(), len_chain=len_chain)
mrw = MCMCSequenceWF(mcmc=RWUncalibrated(), len_chain=len_chain)

print("Running exact MRW...")
samp_mrw = mrw(x=x0_exact, target=target_exact)
mrw_acc_rate = np.mean(samp_mrw.shared['acc_rates'][0])
samp_mrw = samp_mrw.theta['param']
samp_mrw = samp_mrw[burn_in::thinning_step]
print(f"Exact MRW finished. Acc rate: {mrw_acc_rate}")
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
print(f"Cost ratio (target/surrogate) is {(nsamp_pm/(2*nsamp_mcwm))}")
print("Simulation finished.")


if len(sc.ground_truth_par) == 1:
    fig, ax = plt.subplots(figsize=(6, 5))
    sb.histplot(x=samp_pm, stat='density', label='PMMCMC samples', ax=ax)
    sb.histplot(x=samp_mcwm, stat='density', label='MCWM samples', ax=ax)
    ax.legend()
    plt.show()
else:
    # Create DataFrames for both methods
    df_pm = pd.DataFrame(samp_pm, columns=[f'Param {i+1}' for i in range(samp_pm.shape[1])])
    df_mcwm = pd.DataFrame(samp_mcwm, columns=[f'Param {i+1}' for i in range(samp_mcwm.shape[1])])

    # Create base grid using the first dataset
    g = sb.PairGrid(df_pm, diag_sharey=False, corner=False)

    # --- Blue: pseudo-marginal samples ---
    g.map_upper(sb.scatterplot, color="C0", alpha=0.2, s=15)
    g.map_lower(sb.scatterplot, color="C0", alpha=0.2, s=15)
    g.map_upper(sb.kdeplot, color="C0", levels=5)
    g.map_lower(sb.kdeplot, color="C0", levels=5)

    # --- Orange: MCWM samples ---
    def overlay_scatter(x, y, **kwargs):
        sb.scatterplot(x=df_mcwm[x.name], y=df_mcwm[y.name], color="C1", alpha=0.2, s=15, ax=plt.gca())

    def overlay_kde(x, y, **kwargs):
        sb.kdeplot(x=df_mcwm[x.name], y=df_mcwm[y.name], color="C1", levels=5, ax=plt.gca())

    g.map_upper(overlay_scatter)
    g.map_lower(overlay_scatter)
    g.map_upper(overlay_kde)
    g.map_lower(overlay_kde)

    # Diagonal: marginal KDEs
    g.map_diag(sb.kdeplot, color="C0", label="Pseudo-marginal")
    for ax, col in zip(g.diag_axes, df_pm.columns):
        sb.kdeplot(df_mcwm[col], color="C1", label="MCWM", ax=ax)

    # Add legend manually
    handles = [
        plt.Line2D([], [], color="C0", marker="o", linestyle="", label="Pseudo-marginal"),
        plt.Line2D([], [], color="C1", marker="o", linestyle="", label="MCWM")
    ]
    g.figure.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.show()
