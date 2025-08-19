import numpy as np
import scipy.stats as stats
from particles import distributions as dists
from particles import smc_samplers as ssp
from core.samplers import TempAdaptNsampNoRew
from particles.core import SMC
from problem_setup.heat_eq_ip.problem_parameters import SamplerConfig as sc
from problem_setup.heat_eq_ip.likelihood_model import IPModelExact, IPModelInexactFix, IPModelInexactAdapt
from problem_setup.heat_eq_ip.likelihood_model import HeatEqModel as ModelExact
# from problem_setup.toy_problem.problem_parameters import SamplerConfig as sc
# from problem_setup.toy_problem.likelihood_model import IPModelExact, IPModelInexactFix, IPModelInexactAdapt
# from problem_setup.toy_problem.likelihood_model import MatVecModel as ModelExact
import utils.logging as logging
from matplotlib import pyplot as plt
import seaborn as sb
import time

logger = logging.create_custom_logger("smc_logger")
np.random.seed(1111)

if __name__ == "__main__":

    prior = dists.StructDist({'param': dists.Normal(loc=0., 
                                                    scale=sc.prior_marg_var)})

    # Generate synthetic data using ground truth
    true_sol = ModelExact.exact_sol(sc.ground_truth_par).reshape(-1,1)
    #true_sol = ModelExact.exact_binned_sol(sc.ground_truth_par)
    observed_data = true_sol + np.random.normal(loc=0., 
                                                scale=np.sqrt(sc.data_noise_var_seq[0]), 
                                                size=true_sol.shape)

    model = IPModelExact(prior=prior, 
                               data_noise_cov=sc.data_noise_var_seq[0], 
                               data=observed_data)
    # model = IPModelInexactFix(prior=prior, 
    #                                   data_noise_cov=sc.data_noise_var_seq[0], 
    #                                   data=observed_data,
    #                                   nsamp=1000)
    # model = IPModelInexactAdapt(prior=prior,
    #                             data_noise_cov=sc.data_noise_var_seq[0],
    #                             data=observed_data)

    fk_tempering = ssp.AdaptiveTempering(model=model, wastefree=False, len_chain=sc.len_chain, ESSrmin=0.7)
    #fk_tempering = TempAdaptNsampNoRew(Nmin=100, ar_tol=0.35, model=model, wastefree=False, len_chain=sc.len_chain, ESSrmin=0.7)

    my_temp_alg = SMC(fk=fk_tempering, N=sc.n_particles, ESSrmin=1., 
                                verbose=True)
    start_time = time.time()
    my_temp_alg.run()
    my_temp_alg.setup_auxiliary_weights()
    my_temp_alg.resample_move()
    end_time = time.time()
    print(f"Runtime: {end_time-start_time}")
    #print(f"Number of fs samples: {fk_tempering.solver_nsamp}")

    # Define prior and posterior distributions
    prior_dist = stats.norm(loc=0, scale=sc.prior_marg_var)

    # Plot results
    # Create a single subplot instead of (1,2)
    fig, ax = plt.subplots(figsize=(6, 5))  # Remove (1,2), just one plot

    # Plot histogram of sampled parameters
    sb.histplot(x=my_temp_alg.X.theta['param'], stat='density', label='SMC Samples', ax=ax)

    # Plot prior distribution
    x_vals = np.linspace(-3 * sc.prior_marg_var, 3 * sc.prior_marg_var, 1000)
    ax.plot(x_vals, prior_dist.pdf(x_vals), label='Prior', linestyle='dashed', color='black')

    ax.set_xlabel('param')
    ax.legend()

    plt.show()
