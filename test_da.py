import numpy as np
from particles import distributions as dists
from core.samplers import TempAdaptNsampNoRew
from core.mcmc import DA, NoisyMetropolisCal, TwoStageMCMC, RWRegCal
from particles.smc_samplers import AdaptiveMCMCSequence, ArrayRandomWalk
from particles.core import SMC
import particles.smc_samplers as ssp
# from problem_setup.toy_problem.problem_parameters import SamplerConfig as sc
# from problem_setup.toy_problem.likelihood_model import IPModelInexactAdaptHD
# from problem_setup.toy_problem.likelihood_model import MatVecModel as ModelExact
from problem_setup.wos.problem_parameters import SamplerConfig as sc
from problem_setup.wos.likelihood_model import IPModelExactHD, IPModelInexactFixHD, IPModelInexactAdaptHD
from problem_setup.wos.likelihood_model import WOSHD as ModelExact
import logging
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd
import time

np.random.seed(1111)
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
    observed_data = true_sol + rng_data.normal(loc=0., 
                                                scale=np.sqrt(sc.data_noise_var),
                                                size=true_sol.shape)

    # Set up DA
    len_coarse_chain = 0
    len_fine_chain = 80
    fine_over_coarse_nsamp = 32
    # model_fine = IPModelInexactAdaptHD(prior=prior,
    #                             data_noise_cov=sc.data_noise_var,
    #                             data=observed_data)
    model_fine = IPModelInexactFixHD(prior=prior,
                                     data_noise_cov=sc.data_noise_var,
                                     data=observed_data,
                                     nsamp=sc.Nmin)
    model_coarse = IPModelInexactAdaptHD(prior=prior,
                                         data_noise_cov=sc.data_noise_var,
                                         data=observed_data)
    mcmc_coarse = NoisyMetropolisCal()
    mcmc_fine = RWRegCal(eye_prop=True)
    two_stage_chain = TwoStageMCMC(mcmc1=mcmc_coarse,
                                   mcmc2=mcmc_fine,
                                   len_chain1=len_coarse_chain,
                                   len_chain2=len_fine_chain,
                                   coarse_model=model_coarse,
                                   fine_over_coarse=fine_over_coarse_nsamp)

    fk_tempering = ssp.Tempering(model=model_fine,
                                 wastefree=False,
                                 exponents=[1.],
                                 move=two_stage_chain)
    # fk_tempering = TempAdaptNsampNoRew(Nmin=sc.Nmin,
    #                                    ar_tol=sc.ar_tol,
    #                                    model=model_fine,
    #                                    wastefree=False,
    #                                    ESSrmin=sc.ESS_rmin,
    #                                    move=two_stage_chain)

    my_temp_alg = SMC(fk=fk_tempering, N=sc.n_particles, ESSrmin=1., 
                                verbose=True)
    start_time = time.time()
    my_temp_alg.run()
    if resample_move_end:
        my_temp_alg.X.shared["solver_nsamp"] = [sc.Nmin] # TODO DON'T FORGET TO DELETE THIS (USEFUL ONLY FOR 2stage 1IS STEP TESTING)
        my_temp_alg.setup_auxiliary_weights()
        my_temp_alg.resample_move()
    end_time = time.time()
    print(f"Runtime: {end_time-start_time}")

    post_samp = my_temp_alg.X.theta['param']
    np.savez("data/post_samp.npz", post_samp=my_temp_alg.X.theta['param'])

    # # Create a DataFrame for seaborn
    # df = pd.DataFrame(post_samp, columns=[f'Param {i+1}' for i in range(post_samp.shape[1])])

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