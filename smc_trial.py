# smc_trial.py
import numpy as np
import time
from particles import smc_samplers as ssp
from core.samplers import TempAdaptNsampNoRew
from particles.core import SMC

def run_single_smc_trial(data_noise_var, observed_data, prior, ground_truth, len_chain, Nmin, ar_tol, ESSrmin):
    from problem_setup.heat_eq_ip.likelihood_model import IPModelInexactAdapt, IPModelInexactFix

    np.random.seed()  # Important for Ray workers

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

    my_temp_alg = SMC(fk=tempering, N=100, ESSrmin=1., verbose=False)
    start_time = time.time()
    my_temp_alg.run()
    my_temp_alg.resample_move()
    runtime_adapt = time.time() - start_time

    exponents = my_temp_alg.X.shared['exponents']
    estimated_mean_adapt = np.average(my_temp_alg.X.theta['param'], axis=0)
    mse_adapt = np.linalg.norm(estimated_mean_adapt - ground_truth) ** 2
    nsamp_last = tempering.solver_nsamp[-1]
    n_temp_steps = my_temp_alg.t

    model_inex_fix = IPModelInexactFix(
        prior=prior,
        data_noise_cov=data_noise_var,
        data=observed_data,
        nsamp=nsamp_last
    )

    tempering_fixed = ssp.Tempering(
        model=model_inex_fix,
        wastefree=False,
        exponents=exponents[1:],
        len_chain=len_chain
    )

    temp_fixed_alg = SMC(fk=tempering_fixed, N=100, ESSrmin=1., verbose=False)
    start_time = time.time()
    temp_fixed_alg.run()
    temp_fixed_alg.resample_move()
    runtime_fixed = time.time() - start_time

    estimated_mean_fixed = np.average(temp_fixed_alg.X.theta['param'], axis=0)
    mse_fixed = np.linalg.norm(estimated_mean_fixed - ground_truth) ** 2

    return (runtime_adapt, runtime_fixed, mse_adapt, mse_fixed, nsamp_last, n_temp_steps)
