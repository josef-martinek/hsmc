import numpy as np
import scipy.stats as stats
from particles import distributions as dists
from particles import smc_samplers as ssp
from core.samplers import TempAdaptNsampNoRew
from particles.core import SMC
import core.mcmc as MCMC
# from problem_setup.toy_problem.problem_parameters import SamplerConfig as sc
# from problem_setup.toy_problem.likelihood_model import IPModelExactHD, IPModelInexactFixHD, IPModelInexactAdaptHD
# from problem_setup.toy_problem.likelihood_model import MatVecModel as ModelExact
# from problem_setup.fokker_planck.problem_parameters import SamplerConfig as sc
# from problem_setup.fokker_planck.likelihood_model import IPModelExactHD, IPModelInexactFixHD, IPModelInexactAdaptHD
# from problem_setup.fokker_planck.likelihood_model import FPModelHD as ModelExact
from problem_setup.wos.problem_parameters import SamplerConfig as sc
from problem_setup.wos.likelihood_model import IPModelExactHD, IPModelInexactFixHD, IPModelInexactAdaptHD
from problem_setup.wos.likelihood_model import WOSHD as ModelExact
import logging
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd
import time

np.random.seed(1111)
thinning_step = 4
resample_move_end = True
if not resample_move_end:
    logging.error("Posterior pair plot requires unweighted samples")

if __name__ == "__main__":
    if len(sc.ground_truth_par) == 1:
        logging.error("This script is only for multi dimensional inference.")

    param_dim = sc.param_dim
    prior = dists.StructDist({'param': dists.MvNormal(loc=np.zeros(param_dim),
                                                      cov=sc.prior_marg_var*np.eye(param_dim))})

    # Generate synthetic data using ground truth
    true_sol = ModelExact.exact_sol(sc.ground_truth_par).reshape(-1,1)
    rng_data = np.random.default_rng(seed=81)
    observed_data = true_sol + rng_data.normal(loc=0., scale=np.sqrt(sc.data_noise_var), size=true_sol.shape)
    print("Observed data generated.")

    model = IPModelExactHD(prior=prior, 
                               data_noise_cov=sc.data_noise_var, 
                               data=observed_data)
    # model = IPModelInexactFixHD(prior=prior, 
    #                                   data_noise_cov=sc.data_noise_var, 
    #                                   data=observed_data,
    #                                   nsamp=128)
    # model = IPModelInexactAdaptHD(prior=prior,
    #                             data_noise_cov=sc.data_noise_var,
    #                             data=observed_data)

    mcmc = MCMC.RWRegCal()
    move = ssp.AdaptiveMCMCSequence(mcmc=mcmc, adaptive=False, len_chain=sc.len_chain)
    fk_tempering = ssp.AdaptiveTempering(model=model, wastefree=False, ESSrmin=sc.ESS_rmin, move=move)
    #fk_tempering = ssp.AdaptiveTempering(model=model, wastefree=False, len_chain=sc.len_chain, ESSrmin=sc.ESS_rmin)
    #fk_tempering = TempAdaptNsampNoRew(Nmin=sc.Nmin, ar_tol=sc.ar_tol, model=model, wastefree=False, ESSrmin=sc.ESS_rmin, move=move)
    #fk_tempering = TempAdaptNsampNoRew(Nmin=sc.Nmin, ar_tol=sc.ar_tol, model=model, wastefree=False, len_chain=sc.len_chain, ESSrmin=sc.ESS_rmin)

    my_temp_alg = SMC(fk=fk_tempering, N=sc.n_particles, ESSrmin=1., 
                                verbose=True)
    start_time = time.time()
    my_temp_alg.run()
    if resample_move_end:
        my_temp_alg.setup_auxiliary_weights()
        my_temp_alg.resample_move()
    end_time = time.time()
    print(f"Runtime: {end_time-start_time}")

    post_samp = my_temp_alg.X.theta['param']
    post_mean = np.mean(post_samp, axis=0)
    #print("Posterior mean:", post_mean)
    np.savez("data/post_mean_ref.npz", post_mean=post_mean)
    np.savez("data/post_samp_ref.npz", post_samp=my_temp_alg.X.theta['param'])
    post_samp_thinned = post_samp[:,::thinning_step]

    # # Create a DataFrame for seaborn
    # df = pd.DataFrame(post_samp_thinned, columns=[f'Param {thinning_step * i+1}' for i in range(post_samp_thinned.shape[1])])

    # # Create pair plot
    # g = sb.pairplot(df, diag_kind='kde', corner=False,
    #                 plot_kws={'alpha': 0.5, 's': 15},  # small transparent points
    #                 height=2.5)

    # # Optionally, overlay contour density estimates
    # g.map_upper(sb.kdeplot, levels=5, color="C1")
    # g.map_lower(sb.kdeplot, cmap="Blues_d", levels=5)

    # # Adjust layout and show
    # plt.tight_layout()
    # plt.show()