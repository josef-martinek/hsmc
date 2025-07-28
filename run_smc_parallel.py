import ray
import numpy as np
import os
import time
import shutil
from particles import distributions as dists
from problem_setup.heat_eq_ip.problem_parameters import SamplerConfig as sc
from problem_setup.heat_eq_ip.likelihood_model import HeatEqModel as ModelExact
from smc_trial import run_single_smc_trial
import utils.export_info as info

results_folder = "data/smc_results_test"
src = __file__
os.makedirs(results_folder, exist_ok=True)
info.export_full_conda_env(target_folder=results_folder)
info.copy_problem_setup(target_folder=results_folder, module=sc)
info.export_pip_requirements(target_folder=results_folder)
info.export_git_commit(target_folder=results_folder)
info.copy_main_script(target_folder=results_folder, src=src)

ray.init(ignore_reinit_error=True, num_cpus=2)
np.random.seed(1111)

N_runs = 2
Nmin = 100
ar_tol = 0.35
len_chain = 10
gt = sc.ground_truth_par

prior = dists.StructDist({'param': dists.Normal(loc=0., scale=sc.prior_marg_var)})
true_sol = ModelExact.exact_binned_sol(gt)

# Storage
avg_runtimes_adapt = []
avg_runtimes_fixed = []
mse_values_adapt = []
mse_values_fixed = []
mse_stddev_adapt = []
mse_stddev_fixed = []
avg_nsamps_adapt = []
avg_n_temp_steps_adapt = []

@ray.remote
def wrapped_trial(*args):
    return run_single_smc_trial(*args)

for data_noise_var in sc.data_noise_var_seq:
    print(f"Starting noise var {data_noise_var}")
    observed_data = true_sol + np.random.normal(loc=0., scale=np.sqrt(data_noise_var), size=true_sol.shape)
    start_time = time.time()

    tasks = [
        wrapped_trial.remote(data_noise_var, observed_data, prior, gt, len_chain, Nmin, ar_tol, sc.ESS_rmin)
        for _ in range(N_runs)
    ]
    results = ray.get(tasks)
    print(f"Total runtime for noise var={data_noise_var}: {time.time()-start_time}")

    runtimes_adapt, runtimes_fixed, mse_adapt, mse_fixed, nsamps, temp_steps = zip(*results)

    avg_runtimes_adapt.append(np.mean(runtimes_adapt))
    avg_runtimes_fixed.append(np.mean(runtimes_fixed))

    mse_values_adapt.append(np.mean(mse_adapt))
    mse_values_fixed.append(np.mean(mse_fixed))

    mse_stddev_adapt.append(np.std(mse_adapt, ddof=1))
    mse_stddev_fixed.append(np.std(mse_fixed, ddof=1))

    avg_nsamps_adapt.append(np.mean(nsamps))
    avg_n_temp_steps_adapt.append(np.mean(temp_steps))
    print(f"Average number of tempering steps for noise var {data_noise_var} is {avg_n_temp_steps_adapt[-1]}\n")

# 95% CI
mse_se_adapt = np.array(mse_stddev_adapt) / np.sqrt(N_runs)
mse_lower_adapt = np.array(mse_values_adapt) - 1.96 * mse_se_adapt
mse_upper_adapt = np.array(mse_values_adapt) + 1.96 * mse_se_adapt

mse_se_fixed = np.array(mse_stddev_fixed) / np.sqrt(N_runs)
mse_lower_fixed = np.array(mse_values_fixed) - 1.96 * mse_se_fixed
mse_upper_fixed = np.array(mse_values_fixed) + 1.96 * mse_se_fixed

# Save
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

# Save script
current_script = __file__
shutil.copy(current_script, os.path.join(results_folder, "stoch_solver_vs_vanilla_smc.py"))
