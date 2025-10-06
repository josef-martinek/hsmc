# smc_trial.py
import numpy as np
import time
from particles import smc_samplers as ssp
from core.samplers import TempAdaptNsampNoRew
from particles.core import SMC
import logging

def run_single_smc_trial(data_noise_var, true_sol, prior, ground_truth, len_chain, Nmin, ar_tol, ESSrmin, nparticles, resample_move_end, run_id=None, observed_data=None):
    #from problem_setup.heat_eq_ip.likelihood_model import IPModelInexactAdapt, IPModelInexactFix, IPModelExact
    from problem_setup.toy_problem.likelihood_model import IPModelExact, IPModelInexactFix, IPModelInexactAdapt

    if run_id is not None:
        np.random.seed(seed=run_id)  # Important for Ray workers
    if observed_data is None:
        observed_data = true_sol + np.random.normal(loc=0., scale=np.sqrt(data_noise_var), size=true_sol.shape)

    model_inex_adapt = IPModelInexactAdapt(
        prior=prior,
        data_noise_cov=data_noise_var,
        data=observed_data
    )

    tempering = TempAdaptNsampNoRew(
        Nmin=Nmin,
        ar_tol=ar_tol,
        model=model_inex_adapt,
        wastefree=False,
        len_chain=len_chain,
        ESSrmin=ESSrmin
    )

    my_temp_alg = SMC(fk=tempering, N=nparticles, ESSrmin=1., verbose=False)
    start_time = time.time()
    my_temp_alg.run()
    if resample_move_end:
        my_temp_alg.setup_auxiliary_weights()
        my_temp_alg.resample_move()
    runtime_ansamp = time.time() - start_time

    exponents = my_temp_alg.X.shared['exponents']
    if resample_move_end:
        estimated_mean_adapt = np.average(my_temp_alg.X.theta['param'], axis=0)
    else:
        estimated_mean_adapt = np.average(my_temp_alg.X.theta['param'], weights=my_temp_alg.W, axis=0)
    mse_ansamp = np.linalg.norm(estimated_mean_adapt - ground_truth) ** 2
    nsamp_last_ansamp = tempering.solver_nsamp[-1]
    n_temp_steps_ansamp = my_temp_alg.t
    
    if my_temp_alg.t != len(tempering.solver_nsamp[1:]):
        logging.warning("Length of tempering exponents and solver nsamps don't match, cost comparison might be unfair.")
        print(f"t={my_temp_alg.t}")
        print(f"solver_nsamp={tempering.solver_nsamp}")
    if resample_move_end:
        sim_cost_ansamp = sum(tempering.solver_nsamp[1:]) #Cost proportional to number of MCMC steps, neglect cost of IS
        sim_cost_fnsamp_ftemp = nsamp_last_ansamp * my_temp_alg.t
    else:
        sim_cost_ansamp = sum(tempering.solver_nsamp[1:-1])
        sim_cost_fnsamp_ftemp = nsamp_last_ansamp * (my_temp_alg.t-1)

    model_inex_fix = IPModelInexactFix(
        prior=prior,
        data_noise_cov=data_noise_var,
        data=observed_data,
        nsamp=nsamp_last_ansamp
    )

    tempering_fixed = ssp.Tempering(
        model=model_inex_fix,
        wastefree=False,
        exponents=exponents[1:],
        len_chain=len_chain
    )

    temp_fixed_alg = SMC(fk=tempering_fixed, N=nparticles, ESSrmin=1., verbose=False)
    start_time = time.time()
    temp_fixed_alg.run()
    if resample_move_end:
        temp_fixed_alg.setup_auxiliary_weights()
        temp_fixed_alg.resample_move()
    runtime_fnsamp_ftemp = time.time() - start_time

    if resample_move_end:
        estimated_mean_fixed = np.average(temp_fixed_alg.X.theta['param'], axis=0)
    else:
        estimated_mean_fixed = np.average(temp_fixed_alg.X.theta['param'], weights=temp_fixed_alg.W, axis=0)
    mse_fnsamp_ftemp = np.linalg.norm(estimated_mean_fixed - ground_truth) ** 2

    model_fnsamp_atemp = IPModelInexactFix(
        prior=prior,
        data_noise_cov=data_noise_var,
        data=observed_data,
        nsamp=nsamp_last_ansamp
    )

    tempering = ssp.AdaptiveTempering(
        model=model_fnsamp_atemp,
        wastefree=False,
        len_chain=len_chain,
        ESSrmin=ESSrmin
    )

    temp_alg = SMC(fk=tempering, N=nparticles, ESSrmin=1., verbose=False)
    start_time = time.time()
    temp_alg.run()
    if resample_move_end:
        temp_alg.setup_auxiliary_weights()
        temp_alg.resample_move()
    runtime_fnsamp_atemp = time.time() - start_time
    if resample_move_end:
        estimated_mean_fnsamp_atemp = np.average(temp_alg.X.theta['param'], axis=0)
    else:
        estimated_mean_fnsamp_atemp = np.average(temp_alg.X.theta['param'], weights=temp_alg.W, axis=0)
    mse_fnsamp_atemp = np.linalg.norm(estimated_mean_fnsamp_atemp - ground_truth)**2
    if resample_move_end:
        sim_cost_fnsamp_atemp = nsamp_last_ansamp * temp_alg.t
    else:
        sim_cost_fnsamp_atemp = nsamp_last_ansamp * (temp_alg.t-1)
    n_temp_steps_fsamp_atemp = temp_alg.t


    model_exact = IPModelExact(
        prior=prior,
        data_noise_cov=data_noise_var,
        data=observed_data
    )

    tempering = ssp.AdaptiveTempering(
        model=model_exact,
        wastefree=False,
        len_chain=len_chain,
        ESSrmin=ESSrmin
    )

    temp_alg = SMC(fk=tempering, N=nparticles, ESSrmin=1., verbose=False)
    start_time = time.time()
    temp_alg.run()
    if resample_move_end:
        temp_alg.setup_auxiliary_weights()
        temp_alg.resample_move()
    runtime_exact = time.time() - start_time
    if resample_move_end:
        estimated_mean_fnsamp_atemp = np.average(temp_alg.X.theta['param'], axis=0)
    else:
        estimated_mean_fnsamp_atemp = np.average(temp_alg.X.theta['param'], weights=temp_alg.W, axis=0)
    mse_exact = np.linalg.norm(estimated_mean_fnsamp_atemp - ground_truth)**2
    n_temp_steps_exact = temp_alg.t

    return (runtime_ansamp, runtime_fnsamp_ftemp, runtime_fnsamp_atemp, runtime_exact, 
            mse_ansamp, mse_fnsamp_ftemp, mse_fnsamp_atemp, mse_exact,
            sim_cost_ansamp, sim_cost_fnsamp_ftemp, sim_cost_fnsamp_atemp,
            nsamp_last_ansamp, n_temp_steps_ansamp, n_temp_steps_fsamp_atemp, n_temp_steps_exact)
