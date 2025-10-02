import numpy as np
import scipy.stats as stats
from particles import distributions as dists
from particles import smc_samplers as ssp
from core.samplers import TempAdaptNsampNoRew
from particles.core import SMC
from problem_setup.rayleigh_problem_dsmc.problem_parameters import SamplerConfig as sc
from problem_setup.rayleigh_problem_dsmc.likelihood_model import IPModelInexactFix, IPModelInexactAdapt
import logging
from matplotlib import pyplot as plt
import seaborn as sb
import time

np.random.seed(1111)
resample_move_end = True
if resample_move_end is False:
    logging.error("simulation supported only with moving at the end.")

if __name__ == "__main__":

    prior = dists.StructDist({'param': dists.Normal(loc=sc.prior_mean, 
                                                    scale=sc.prior_marg_var)})

    # Generate synthetic data using ground truth
    true_data = np.load("problem_setup/rayleigh_problem_dsmc/ground_truth/dsmc_data.npz")
    true_sol = true_data["uy"]
    observed_data = true_sol + np.random.normal(loc=0., 
                                                scale=np.sqrt(sc.data_noise_var), 
                                                size=true_sol.shape)
    data_reshaped = observed_data.reshape(-1,1) #broadcast from (n,) to (n,1)

    # model = IPModelInexactFix(prior=prior, 
    #                                   data_noise_cov=sc.data_noise_var_seq[0], 
    #                                   data=observed_data,
    #                                   nsamp=1000)
    model = IPModelInexactAdapt(prior=prior,
                                data_noise_cov=sc.data_noise_var,
                                data=data_reshaped)

    # fk_tempering = ssp.AdaptiveTempering(model=model, wastefree=False, len_chain=sc.len_chain, ESSrmin=sc.ESS_rmin)
    fk_tempering = TempAdaptNsampNoRew(Nmin=sc.Nmin, ar_tol=sc.ar_tol, model=model, wastefree=False, len_chain=sc.len_chain, ESSrmin=sc.ESS_rmin)

    my_temp_alg = SMC(fk=fk_tempering, N=sc.n_particles, ESSrmin=1., 
                                verbose=True)
    start_time = time.time()
    my_temp_alg.run()
    if resample_move_end:
        my_temp_alg.setup_auxiliary_weights()
        my_temp_alg.resample_move()
    end_time = time.time()
    print(f"Runtime: {end_time-start_time}")
    #print(f"Number of fs samples: {fk_tempering.solver_nsamp}")
    np.savez("dsmc_infer_results", post_samp=my_temp_alg.X.theta['param'],
             prior_mean=sc.prior_mean,
             prior_marg_var=sc.prior_marg_var,
             true_sol=true_sol,
             noisy_data=observed_data)
