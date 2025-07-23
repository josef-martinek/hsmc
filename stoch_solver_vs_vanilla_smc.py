import os
import numpy as np
import scipy.stats as stats
import time
import matplotlib.pyplot as plt
import seaborn as sb
from particles import distributions as dists
from particles import smc_samplers as ssp
from core.samplers import TemperingAdaptNsamp2
from particles.core import SMC
from problem_parameters import MRWConfig as SamplerConfig
import problem_parameters as params
import utils.logging as logg
from smc_stoch_solver_tempering import IPModelExact, IPModelInexact
from smc_linear_bip import IPModelInexact as IPModelInexactFixed
import shutil

logger = logg.create_custom_logger("smc_logger")
np.random.seed(1111)

# Define results folder
results_folder = "data/smc_results_test"
os.makedirs(results_folder, exist_ok=True)  # Create folder if it doesn't exist

# SMC settings
Nmin = 100
ar_tol = 0.35
N_runs = 10
len_chain = 10

#ESS_rmin_seq = [0.9, 0.95, 0.975]
#ESS_rmin_seq = [0.9, 0.9625, 0.9859375, 0.9947265625]
#ESS_rmin_seq = [0.9, 0.99, 0.999]
#ESS_rmin_seq = np.logspace(start=np.log10(0.9), stop=np.log10(0.999), num=6, endpoint=True, base=10)

# num_points = 4
# epsilons = np.logspace(-1, -3, num=num_points)
# ESS_rmin_seq = 1 - epsilons
# print(f"ESS_min sequence is: {ESS_rmin_seq}")
ESS_rmin_seq = [0.7]

# Problem setup
sc = SamplerConfig()
parameterDim = len(sc.init_state)

prior = dists.StructDist({'param': dists.Normal(loc=0., scale=sc.prior_marg_var)})
exact_fw_op = params.FwProblemExact().exact_fw_operator

# Generate synthetic data
observed_data = exact_fw_op @ (sc.ground_truth_par * np.ones((exact_fw_op.shape[1], 1))) + \
                np.random.normal(loc=0., scale=np.sqrt(sc.data_noise_var), size=exact_fw_op.shape)

# Define models
exact_model = IPModelExact(fw_operator=exact_fw_op, prior=prior, data_noise_var=sc.data_noise_var, data=observed_data)
inexact_model = IPModelInexact(fw_model=params.FwProblemInexactAdapt(), prior=prior, data_noise_var=sc.data_noise_var, data=observed_data)

# Compute true posterior parameters
A = exact_fw_op
Gamma_inv = np.linalg.inv(sc.data_noise_var * np.eye(A.shape[0]))  # Inverse noise covariance
C0_inv = np.linalg.inv(np.eye(parameterDim) * sc.prior_marg_var)  # Inverse prior covariance
C_inv = A.T @ Gamma_inv @ A + C0_inv
C = np.linalg.inv(C_inv)  # Posterior covariance
m = C @ (A.T @ Gamma_inv @ observed_data + C0_inv @ np.zeros(parameterDim))  # Posterior mean

# Initialize storage
avg_runtimes_adapt = []
avg_runtimes_fixed = []
mse_values_adapt = []
mse_values_fixed = []
mse_stddev_adapt = []  # Store standard deviation of MSE
mse_stddev_fixed = []
avg_nsamps_adapt = []
avg_n_temp_steps_adapt = []

# Run SMC for different ESS_rmin values
for ESS_rmin in ESS_rmin_seq:
    runtimes_fixed = []
    runtimes_adapt = []
    mse_per_run_adapt = []
    mse_per_run_fixed = []
    number_of_fw_solver_samples_adapt = []
    n_temp_steps_adapt = []
    print_time_start = time.time()

    for _ in range(N_runs):
        tempering = TemperingAdaptNsamp2(Nmin=Nmin, ar_tol=ar_tol, 
                                                model=inexact_model, wastefree=False, len_chain=len_chain, ESSrmin=ESS_rmin)
        
        my_temp_alg = SMC(fk=tempering, N=1000, ESSrmin=1., verbose=False)
        start_time = time.time()
        my_temp_alg.run()
        my_temp_alg.resample_move()
        runtime = time.time() - start_time
        runtimes_adapt.append(runtime)

        # Compute posterior mean from SMC samples
        exponents = my_temp_alg.X.shared['exponents']
        #estimated_mean = np.average(my_temp_alg.X.theta['param'], weights=my_temp_alg.W, axis=0)
        estimated_mean = np.average(my_temp_alg.X.theta['param'], axis=0)

        # Compute squared error
        mse_per_run_adapt.append(np.linalg.norm(estimated_mean - m) ** 2)
        number_of_fw_solver_samples_adapt.append(tempering.solver_nsamp[-1])
        n_temp_steps_adapt.append(my_temp_alg.t)

        inexact_model_fixed = IPModelInexactFixed(fw_model=params.FwProblemInexact(n_particles=tempering.solver_nsamp[-1]), 
                                                  prior=prior, 
                                                  data_noise_var=sc.data_noise_var, 
                                                  data=observed_data)
        tempering_fixed = ssp.Tempering(model=inexact_model_fixed,
                                        wastefree=False,
                                        exponents=exponents[1:],
                                        len_chain=len_chain)
        temp_fixed_alg = SMC(fk=tempering_fixed, N=1000, ESSrmin=1., verbose=False)
        start_time = time.time()
        temp_fixed_alg.run()
        temp_fixed_alg.resample_move()
        runtime = time.time() - start_time
        runtimes_fixed.append(runtime)

        #estimated_mean = np.average(temp_fixed_alg.X.theta['param'], weights=temp_fixed_alg.W, axis=0)
        estimated_mean = np.average(temp_fixed_alg.X.theta['param'], axis=0)
        # Compute squared error
        mse_per_run_fixed.append(np.linalg.norm(estimated_mean - m) ** 2)
    
    print(f"Total runtime for ESS_rmin={ESS_rmin}: {time.time()-print_time_start}")

    avg_n_temp_steps_adapt.append(np.mean(n_temp_steps_adapt))
    print(f"Average number of tempering steps for ESS_min {ESS_rmin} is {avg_n_temp_steps_adapt[-1]}")
    
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
    print(f"Average solver nsamp in the last SMC step for ESS_rmin {ESS_rmin} is {avg_solver_nsamp}")

# Compute 95% confidence intervals
mse_se_adapt = np.array(mse_stddev_adapt) / np.sqrt(N_runs)  # Standard error
mse_lower_adapt = np.array(mse_values_adapt) - 1.96 * mse_se_adapt
mse_upper_adapt = np.array(mse_values_adapt) + 1.96 * mse_se_adapt

mse_se_fixed = np.array(mse_stddev_fixed) / np.sqrt(N_runs)  # Standard error
mse_lower_fixed = np.array(mse_values_fixed) - 1.96 * mse_se_fixed
mse_upper_fixed = np.array(mse_values_fixed) + 1.96 * mse_se_fixed

# Save numerical results
np.savez(os.path.join(results_folder, "smc_results.npz"),
         ESS_rmin_seq=ESS_rmin_seq,
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
