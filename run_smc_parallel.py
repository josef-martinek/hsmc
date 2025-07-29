# smc_run_parallel_allvars.py
import os
import time
import shutil
from collections import defaultdict

import numpy as np
import ray

from particles import distributions as dists

from problem_setup.heat_eq_ip.problem_parameters import SamplerConfig as sc
from problem_setup.heat_eq_ip.likelihood_model import HeatEqModel as ModelExact
from smc_trial import run_single_smc_trial
import smc_trial
import utils.export_info as info


# -----------------------------
# Config
# -----------------------------
RESULTS_FOLDER = "data/smc_results_test"   # or info.create_results_folder()
#NUM_CPUS = int(os.environ.get("RAY_NUM_CPUS", "0"))  # 0 -> let Ray autodetect
NUM_CPUS = 2
SEED = 1111

# SMC hyperparameters
N_runs = 2
Nmin = 100
ar_tol = 0.35
len_chain = 10

# -----------------------------
# Housekeeping / provenance
# -----------------------------
os.makedirs(RESULTS_FOLDER, exist_ok=True)
src = __file__
np.random.seed(SEED)

# Export reproducibility info
info.export_full_conda_env(target_folder=RESULTS_FOLDER)
info.copy_problem_setup(target_folder=RESULTS_FOLDER, module=sc)
info.copy_problem_setup(target_folder=RESULTS_FOLDER, module=smc_trial)
info.copy_problem_setup(target_folder=RESULTS_FOLDER, module=ModelExact)
info.export_pip_requirements(target_folder=RESULTS_FOLDER)
info.export_git_commit(target_folder=RESULTS_FOLDER)
info.copy_main_script(target_folder=RESULTS_FOLDER, src=src)

# -----------------------------
# Ray init
# -----------------------------
if NUM_CPUS > 0:
    ray.init(ignore_reinit_error=True, num_cpus=NUM_CPUS)
else:
    ray.init(ignore_reinit_error=True)  # auto-detect


# -----------------------------
# Problem setup
# -----------------------------
prior = dists.StructDist({'param': dists.Normal(loc=0., scale=sc.prior_marg_var)})
gt = sc.ground_truth_par
true_sol = ModelExact.exact_binned_sol(gt)  # fixed "truth" used by all trials

data_noise_var_seq = sc.data_noise_var_seq

# Storage (lists; will save as arrays)
avg_runtimes_ansamp = []
avg_runtimes_fnsamp_ftemp = []
avg_runtimes_fnsamp_atemp = []
avg_runtimes_exact = []

avg_sim_cost_ansamp = []
avg_sim_cost_fnsamp_ftemp = []
avg_sim_cost_fnsamp_atemp = []

mse_values_ansamp = []
mse_values_fnsamp_ftemp = []
mse_values_fnsamp_atemp = []
mse_values_exact = []

mse_stddev_ansamp = []
mse_stddev_fnsamp_ftemp = []
mse_stddev_fnsamp_atemp = []
mse_stddev_exact = []

avg_last_nsamps_ansamp = []
avg_n_temp_steps_ansamp = []
avg_n_temp_steps_fnsamp_atemp = []
avg_n_temp_steps_exact = []


@ray.remote
def wrapped_trial(*args):
    # Simply delegate to your existing runner
    return run_single_smc_trial(*args)


# -----------------------------
# Submit ALL tasks up-front
# -----------------------------
submitted_refs = []
ref_to_noise = {}

print("Submitting all Ray tasks...")
submit_start = time.time()

for noise_var in data_noise_var_seq:
    for run_idx in range(N_runs):
        ref = wrapped_trial.remote(
            noise_var,
            true_sol,
            prior,
            gt,
            len_chain,
            Nmin,
            ar_tol,
            sc.ESS_rmin,
            sc.n_particles,
        )
        submitted_refs.append(ref)
        ref_to_noise[ref] = noise_var

print(f"Submitted {len(submitted_refs)} tasks in {time.time() - submit_start:.2f}s.")

# -----------------------------
# Collect results as they finish
# -----------------------------
results_by_noise = defaultdict(list)

pending = set(submitted_refs)
n_done = 0
total = len(submitted_refs)
t0 = time.time()

print("Collecting results as they finish...")
while pending:
    # Wait for next finished task
    ready, pending = ray.wait(list(pending), num_returns=1, timeout=None)
    ref = ready[0]
    noise_var = ref_to_noise[ref]

    # Get its result and store
    result = ray.get(ref)
    results_by_noise[noise_var].append(result)

    n_done += 1
    if n_done % max(1, total // 20) == 0 or n_done == total:
        elapsed = time.time() - t0
        print(f"[{n_done}/{total}] collected in {elapsed:.1f}s")

print(f"All results collected in {time.time() - t0:.2f}s.")

# -----------------------------
# Reduce to aggregates per noise level (in original order)
# -----------------------------
for noise_var in data_noise_var_seq:
    res_list = results_by_noise[noise_var]
    # Unpack tuples
    (
        runtimes_ansamp,
        runtimes_fnsamp_ftemp,
        runtimes_fnsamp_atemp,
        runtimes_exact,
        mse_ansamp,
        mse_fnsamp_ftemp,
        mse_fnsamp_atemp,
        mse_exact,
        sim_cost_ansamp,
        sim_cost_fnsamp_ftemp,
        sim_cost_fnsamp_atemp,
        nsamps_last_ansamp,
        n_temp_steps_ansamp,
        n_temp_fnsamp_atemp,
        n_temp_exact,
    ) = zip(*res_list)

    # Means
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

    # Std dev across runs (ddof=1)
    mse_stddev_ansamp.append(np.std(mse_ansamp, ddof=1))
    mse_stddev_fnsamp_ftemp.append(np.std(mse_fnsamp_ftemp, ddof=1))
    mse_stddev_fnsamp_atemp.append(np.std(mse_fnsamp_atemp, ddof=1))
    mse_stddev_exact.append(np.std(mse_exact, ddof=1))

    avg_last_nsamps_ansamp.append(np.mean(nsamps_last_ansamp))
    avg_n_temp_steps_ansamp.append(np.mean(n_temp_steps_ansamp))
    avg_n_temp_steps_fnsamp_atemp.append(np.mean(n_temp_fnsamp_atemp))
    avg_n_temp_steps_exact.append(np.mean(n_temp_exact))

# -----------------------------
# Save everything in your format
# -----------------------------
np.savez(
    os.path.join(RESULTS_FOLDER, "smc_results.npz"),
    data_noise_var_seq=data_noise_var_seq,
    n_runs=N_runs,

    avg_runtimes_ansamp=np.array(avg_runtimes_ansamp),
    avg_runtimes_fnsamp_ftemp=np.array(avg_runtimes_fnsamp_ftemp),
    avg_runtimes_fnsamp_atemp=np.array(avg_runtimes_fnsamp_atemp),
    avg_runtimes_exact=np.array(avg_runtimes_exact),

    avg_sim_cost_ansamp=np.array(avg_sim_cost_ansamp),
    avg_sim_cost_fnsamp_ftemp=np.array(avg_sim_cost_fnsamp_ftemp),
    avg_sim_cost_fnsamp_atemp=np.array(avg_sim_cost_fnsamp_atemp),

    mse_values_ansamp=np.array(mse_values_ansamp),
    mse_values_fnsamp_ftemp=np.array(mse_values_fnsamp_ftemp),
    mse_values_fnsamp_atemp=np.array(mse_values_fnsamp_atemp),
    mse_values_exact=np.array(mse_values_exact),

    mse_stddev_ansamp=np.array(mse_stddev_ansamp),
    mse_stddev_fnsamp_ftemp=np.array(mse_stddev_fnsamp_ftemp),
    mse_stddev_fnsamp_atemp=np.array(mse_stddev_fnsamp_atemp),
    mse_stddev_exact=np.array(mse_stddev_exact),

    avg_last_nsamps_ansamp=np.array(avg_last_nsamps_ansamp),
    avg_n_temp_steps_ansamp=np.array(avg_n_temp_steps_ansamp),
    avg_n_temp_steps_fnsamp_atemp=np.array(avg_n_temp_steps_fnsamp_atemp),
    avg_n_temp_steps_exact=np.array(avg_n_temp_steps_exact),
)

print(f"\nSaved results to {os.path.join(RESULTS_FOLDER, 'smc_results.npz')}")
