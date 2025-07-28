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

results_folder = info.create_results_folder()
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
avg_runtimes_ansamp, avg_runtimes_fnsamp_ftemp, avg_runtimes_fnsamp_atemp, avg_runtimes_exact = [], [], [], []
avg_sim_cost_ansamp, avg_sim_cost_fnsamp_ftemp, avg_sim_cost_fnsamp_atemp = [], [], []
mse_values_ansamp, mse_values_fnsamp_ftemp, mse_values_fnsamp_atemp, mse_values_exact = [], [], [], []
mse_stddev_ansamp, mse_stddev_fnsamp_ftemp, mse_stddev_fnsamp_atemp, mse_stddev_exact = [], [], [], []
avg_last_nsamps_ansamp, avg_n_temp_steps_ansamp, avg_n_temp_steps_fnsamp_atemp, avg_n_temp_steps_exact = [], [], [], []

@ray.remote
def wrapped_trial(*args):
    return run_single_smc_trial(*args)


for data_noise_var in sc.data_noise_var_seq:
    print(f"Starting noise var {data_noise_var}")
    observed_data = true_sol + np.random.normal(loc=0., scale=np.sqrt(data_noise_var), size=true_sol.shape)
    start_time = time.time()

    tasks = [
        wrapped_trial.remote(data_noise_var, observed_data, prior, gt, len_chain, Nmin, ar_tol, sc.ESS_rmin, sc.n_particles)
        for _ in range(N_runs)
    ]
    results = ray.get(tasks)
    print(f"Total runtime for noise var={data_noise_var}: {time.time()-start_time}")

    (
        runtimes_ansamp, runtimes_fnsamp_ftemp, runtimes_fnsamp_atemp, runtimes_exact,
        mse_ansamp, mse_fnsamp_ftemp, mse_fnsamp_atemp, mse_exact,
        sim_cost_ansamp, sim_cost_fnsamp_ftemp, sim_cost_fnsamp_atemp,
        nsamps_last_ansamp, n_temp_steps_ansamp, n_temp_fnsamp_atemp, n_temp_exact
    ) = zip(*results)

    # Store means
    avg_runtimes_ansamp.append(np.mean(runtimes_ansamp))
    avg_runtimes_fnsamp_ftemp.append(np.mean(runtimes_fnsamp_ftemp))
    avg_runtimes_fnsamp_atemp.append(np.mean(runtimes_fnsamp_atemp))
    avg_runtimes_exact.append(np.mean(runtimes_exact))

    avg_sim_cost_ansamp.append(np.mean(sim_cost_ansamp))
    avg_sim_cost_fnsamp_ftemp.append(np.mean(sim_cost_fnsamp_ftemp))
    avg_sim_cost_fnsamp_atemp.append(np.mean(sim_cost_fnsamp_atemp))

    mse_values_ansamp.append(np.mean(mse_ansamp))
    mse_values_fnsamp_ftemp.append(np.mean(mse_fnsamp_ftemp))
    mse_values_fnsamp_atemp.append(np.mean(mse_fnsamp_atemp))
    mse_values_exact.append(np.mean(mse_exact))

    mse_stddev_ansamp.append(np.std(mse_ansamp, ddof=1))
    mse_stddev_fnsamp_ftemp.append(np.std(mse_fnsamp_ftemp, ddof=1))
    mse_stddev_fnsamp_atemp.append(np.std(mse_fnsamp_atemp, ddof=1))
    mse_stddev_exact.append(np.std(mse_exact, ddof=1))

    avg_last_nsamps_ansamp.append(np.mean(nsamps_last_ansamp))
    avg_n_temp_steps_ansamp.append(np.mean(n_temp_steps_ansamp))
    avg_n_temp_steps_fnsamp_atemp.append(np.mean(n_temp_fnsamp_atemp))
    avg_n_temp_steps_exact.append(np.mean(n_temp_exact))

    print(f"Average tempering steps (adaptive nsamp): {avg_n_temp_steps_ansamp[-1]}\n")

# Save everything
np.savez(os.path.join(results_folder, "smc_results.npz"),
         data_noise_var_seq=sc.data_noise_var_seq,
         n_runs=N_runs,

         avg_runtimes_ansamp=avg_runtimes_ansamp,
         avg_runtimes_fnsamp_ftemp=avg_runtimes_fnsamp_ftemp,
         avg_runtimes_fnsamp_atemp=avg_runtimes_fnsamp_atemp,
         avg_runtimes_exact=avg_runtimes_exact,
 
         avg_sim_cost_ansamp=avg_sim_cost_ansamp,
         avg_sim_cost_fnsamp_ftemp=avg_sim_cost_fnsamp_ftemp,
         avg_sim_cost_fnsamp_atemp=avg_sim_cost_fnsamp_atemp,

         mse_values_ansamp=mse_values_ansamp,
         mse_values_fnsamp_ftemp=mse_values_fnsamp_ftemp,
         mse_values_fnsamp_atemp=mse_values_fnsamp_atemp,
         mse_values_exact=mse_values_exact,

         mse_stddev_ansamp=mse_stddev_ansamp,
         mse_stddev_fnsamp_ftemp=mse_stddev_fnsamp_ftemp,
         mse_stddev_fnsamp_atemp=mse_stddev_fnsamp_atemp,
         mse_stddev_exact=mse_stddev_exact,

         avg_last_nsamps_ansamp=avg_last_nsamps_ansamp,
         avg_n_temp_steps_ansamp=avg_n_temp_steps_ansamp,
         avg_n_temp_steps_fnsamp_atemp=avg_n_temp_steps_fnsamp_atemp,
         avg_n_temp_steps_exact=avg_n_temp_steps_exact
     )