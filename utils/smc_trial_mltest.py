import numpy as np
import time
from particles import smc_samplers as ssp
from core.samplers import TempAdaptNsampNoRew
from particles.core import SMC
import logging

def run_single_smc_trial(data_noise_var, true_sol, prior, ground_truth, chain, Nmin, ar_tol, ESSrmin, nparticles, resample_move_end, run_id=None, observed_data=None):
    #from problem_setup.toy_problem.likelihood_model import IPModelInexactAdaptHD
    #from problem_setup.fokker_planck.likelihood_model import IPModelInexactAdaptHD
    from problem_setup.wos.likelihood_model import IPModelInexactAdaptHD

    if run_id is not None:
        np.random.seed(seed=run_id)  # Important for Ray workers
    if observed_data is None:
        observed_data = true_sol + np.random.normal(loc=0., scale=np.sqrt(data_noise_var), size=true_sol.shape)

    model_inex_adapt = IPModelInexactAdaptHD(
        prior=prior,
        data_noise_cov=data_noise_var,
        data=observed_data
    )

    tempering = TempAdaptNsampNoRew(
        Nmin=Nmin,
        ar_tol=ar_tol,
        model=model_inex_adapt,
        wastefree=False,
        move=chain,
        ESSrmin=ESSrmin
    )

    my_temp_alg = SMC(fk=tempering, N=nparticles, ESSrmin=1., verbose=False)
    start_time = time.time()
    my_temp_alg.run()
    if resample_move_end:
        my_temp_alg.setup_auxiliary_weights()
        my_temp_alg.resample_move()
    runtime_ansamp = time.time() - start_time

    if resample_move_end:
        estimated_mean_ansamp = np.average(my_temp_alg.X.theta['param'], axis=0)
    else:
        estimated_mean_ansamp = np.average(my_temp_alg.X.theta['param'], weights=my_temp_alg.W, axis=0)
    mse_ansamp = np.linalg.norm(estimated_mean_ansamp - ground_truth) ** 2
    nsamp_last_ansamp = tempering.solver_nsamp[-1]
    n_temp_steps_ansamp = my_temp_alg.t
    
    if my_temp_alg.t != len(tempering.solver_nsamp[1:]):
        logging.warning("Length of tempering exponents and solver nsamps don't match, cost comparison might be unfair.")
        print(f"t={my_temp_alg.t}")
        print(f"solver_nsamp={tempering.solver_nsamp}")
    if resample_move_end:
        if chain.len_chain1 == 1:
            sim_cost_ansamp = sum(tempering.solver_nsamp[1:])*((chain.len_chain1-1)*(2/chain.fine_over_coarse)+(chain.len_chain2-1))
        else:
            sim_cost_ansamp = sum(tempering.solver_nsamp[1:])*((chain.len_chain1-1)*(2/chain.fine_over_coarse)+(chain.len_chain2))
        #Cost proportional to number of MCMC steps, neglect cost of IS, works only for TwoStageMCMC
        # Nr of chain steps = length-1. Also MCWM makes two evals per step.
    else:
        if chain.len_chain1 == 1:
            sim_cost_ansamp = sum(tempering.solver_nsamp[1:-1])*((chain.len_chain1-1)*(2/chain.fine_over_coarse)+(chain.len_chain2-1))
        else:
            sim_cost_ansamp = sum(tempering.solver_nsamp[1:-1])*((chain.len_chain1-1)*(2/chain.fine_over_coarse)+(chain.len_chain2))
    sim_cost_ansamp += tempering.solver_nsamp[0] #Cost of Likelihood eval in initialisation
    for i in range(my_temp_alg.t): # Cost of generalised IS
        if tempering.solver_nsamp[i+1]>tempering.solver_nsamp[i]:
            sim_cost_ansamp += tempering.solver_nsamp[i+1]
    sim_cost_ansamp = sim_cost_ansamp * model_inex_adapt.cost_normalising_const()

    return (runtime_ansamp,
            mse_ansamp,
            sim_cost_ansamp,
            nsamp_last_ansamp,
            n_temp_steps_ansamp,
            estimated_mean_ansamp,)
