import os
import numpy as np
import scipy.stats as stats
import time
import matplotlib.pyplot as plt
import seaborn as sb
from particles import distributions as dists
from particles import smc_samplers as ssp
from core.samplers import TempAdaptNsampNoRew
from particles.core import SMC
from problem_setup.heat_eq_ip.problem_parameters import SamplerConfig as sc
from problem_setup.heat_eq_ip.likelihood_model import IPModelExact, IPModelInexactFix, IPModelInexactAdapt
from problem_setup.heat_eq_ip.likelihood_model import HeatEqModel as ModelExact
import utils.logging as logging
import shutil

logger = logging.create_custom_logger("smc_logger")
np.random.seed(1111)

# Define results folder
results_folder = "data/smc_results_test"
os.makedirs(results_folder, exist_ok=True)  # Create folder if it doesn't exist

# SMC settings
Nmin = 100
ar_tol = 0.35
N_runs = 2
len_chain = 10

gt = sc.ground_truth_par

prior = dists.StructDist({'param': dists.Normal(loc=0., scale=sc.prior_marg_var)})
true_sol = ModelExact.exact_binned_sol(sc.ground_truth_par)

# Initialize storage
avg_runtimes_adapt = []
avg_runtimes_fixed = []
mse_values_adapt = []
mse_values_fixed = []
mse_stddev_adapt = []  # Store standard deviation of MSE
mse_stddev_fixed = []
avg_nsamps_adapt = []
avg_n_temp_steps_adapt = []

# Run SMC for different noise var
for data_noise_var in sc.data_noise_var_seq:
    observed_data = true_sol + np.random.normal(loc=0., 
                                            scale=np.sqrt(data_noise_var), 
                                            size=true_sol.shape)

    model_inex_adapt = IPModelInexactAdapt(prior=prior,
                                        data_noise_cov=data_noise_var,
                                        data=observed_data)
    runtimes_fixed = []
    runtimes_adapt = []
    mse_per_run_adapt = []
    mse_per_run_fixed = []
    number_of_fw_solver_samples_adapt = []
    n_temp_steps_adapt = []
    print_time_start = time.time()

    for _ in range(N_runs):
        tempering = TempAdaptNsampNoRew(Nmin=Nmin, 
                                        ar_tol=ar_tol, 
                                        model=model_inex_adapt, 
                                        wastefree=False, 
                                        len_chain=len_chain, 
                                        ESSrmin=sc.ESS_rmin)
        
        my_temp_alg = SMC(fk=tempering, N=100, ESSrmin=1., verbose=False)
        start_time = time.time()
        my_temp_alg.run()
        my_temp_alg.resample_move()
        runtime = time.time() - start_time
        runtimes_adapt.append(runtime)

        # Compute posterior mean from SMC samples
        exponents = my_temp_alg.X.shared['exponents']
        estimated_mean = np.average(my_temp_alg.X.theta['param'], axis=0)

        # Compute squared error
        mse_per_run_adapt.append(np.linalg.norm(estimated_mean - gt) ** 2)
        number_of_fw_solver_samples_adapt.append(tempering.solver_nsamp[-1])
        n_temp_steps_adapt.append(my_temp_alg.t)

        model_inex_fix = IPModelInexactFix(prior=prior, 
                                   data_noise_cov=data_noise_var, 
                                   data=observed_data,
                                   nsamp=tempering.solver_nsamp[-1])
        tempering_fixed = ssp.Tempering(model=model_inex_fix,
                                        wastefree=False,
                                        exponents=exponents[1:],
                                        len_chain=len_chain)
        temp_fixed_alg = SMC(fk=tempering_fixed, N=100, ESSrmin=1., verbose=False)
        start_time = time.time()
        temp_fixed_alg.run()
        temp_fixed_alg.resample_move()
        runtime = time.time() - start_time
        runtimes_fixed.append(runtime)

        #estimated_mean = np.average(temp_fixed_alg.X.theta['param'], weights=temp_fixed_alg.W, axis=0)
        estimated_mean = np.average(temp_fixed_alg.X.theta['param'], axis=0)
        # Compute squared error
        mse_per_run_fixed.append(np.linalg.norm(estimated_mean - gt) ** 2)
    
    print(f"Total runtime for noise var={data_noise_var}: {time.time()-print_time_start}")

    avg_n_temp_steps_adapt.append(np.mean(n_temp_steps_adapt))
    print(f"Average number of tempering steps for noise var {data_noise_var} is {avg_n_temp_steps_adapt[-1]}")
    
    # Compute statistics
    avg_runtime = np.mean(runtimes_adapt)
    avg_runtimes_adapt.append(avg_runtime)

    avg_runtime = np.mean(runtimes_fixed)
    avg_runtimes_fixed.append(avg_runtime)

    mse_mean = np.mean(mse_per_run_adapt)
    mse_values_adapt.append(mse_mean)

    mse_mean = np.mean(mse_per_run_fixed)
    mse_values_fixed.append(mse_mean)

    mse_std = np.std(mse_per_run_adapt, ddof=1)  # Standard deviation
    mse_stddev_adapt.append(mse_std)

    mse_std = np.std(mse_per_run_fixed, ddof=1)
    mse_stddev_fixed.append(mse_std)

    avg_solver_nsamp = np.mean(number_of_fw_solver_samples_adapt)
    avg_nsamps_adapt.append(avg_solver_nsamp)
    print(f"Average solver nsamp in the last SMC step for noise var {data_noise_var} is {avg_solver_nsamp}\n")

# Compute 95% confidence intervals
mse_se_adapt = np.array(mse_stddev_adapt) / np.sqrt(N_runs)  # Standard error
mse_lower_adapt = np.array(mse_values_adapt) - 1.96 * mse_se_adapt
mse_upper_adapt = np.array(mse_values_adapt) + 1.96 * mse_se_adapt

mse_se_fixed = np.array(mse_stddev_fixed) / np.sqrt(N_runs)  # Standard error
mse_lower_fixed = np.array(mse_values_fixed) - 1.96 * mse_se_fixed
mse_upper_fixed = np.array(mse_values_fixed) + 1.96 * mse_se_fixed

# Save numerical results
np.savez(os.path.join(results_folder, "smc_results.npz"),
         data_noise_var_seq=sc.data_noise_var_seq,
         avg_runtimes_adapt=avg_runtimes_adapt,
         mse_values_adapt=mse_values_adapt,
         mse_stddev_adapt=mse_stddev_adapt,
         avg_nsamps_adapt=avg_nsamps_adapt,
         mse_lower=mse_lower_adapt,
         mse_upper=mse_upper_adapt,
         avg_runtimes_fixed=avg_runtimes_fixed,
         mse_values_fixed=mse_values_fixed,
         mse_stddev_fixed=mse_stddev_fixed,
         mse_lower_fixed=mse_lower_fixed,
         mse_upper_fixed=mse_upper_fixed)

current_script = __file__  # Get the name of the currently running script
shutil.copy(current_script, os.path.join(results_folder, "stoch_solver_vs_vanilla_smc.py"))
