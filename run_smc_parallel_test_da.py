import os
import time
from collections import defaultdict
import numpy as np
import ray
from particles import distributions as dists

from problem_setup.toy_problem.problem_parameters_mltest import SamplerConfig as sc
from problem_setup.toy_problem.likelihood_model import MatVecModel as ModelExact
import utils.export_info as info
import utils
from utils.create_test_chains import get_chain_seq
import json

if len(sc.ground_truth_par) == 1:
    raise NotImplementedError
else:
    from utils.smc_trial_mltest import run_single_smc_trial

# -----------------------------
# Config
# -----------------------------
RESULTS_FOLDER = "data/smc_results_test"   # or info.create_results_folder()
#RESULTS_FOLDER = info.create_results_folder()
NUM_CPUS = 0
SEED = 1111
resample_move_end = True

# Nr of inference runs for each param in param_seq
N_runs = 2

# -----------------------------
# Housekeeping / provenance
# -----------------------------
os.makedirs(RESULTS_FOLDER, exist_ok=True)
src = __file__
np.random.seed(SEED)

# Export reproducibility info
info.export_full_conda_env(target_folder=RESULTS_FOLDER)
info.copy_problem_setup(target_folder=RESULTS_FOLDER, module=sc)
info.copy_problem_setup(target_folder=RESULTS_FOLDER, module=utils.smc_trial_mltest)
info.copy_problem_setup(target_folder=RESULTS_FOLDER, module=ModelExact)
info.export_pip_requirements(target_folder=RESULTS_FOLDER)
info.export_git_commit(target_folder=RESULTS_FOLDER)
info.copy_main_script(target_folder=RESULTS_FOLDER, src=src)
#TODO save chain setup

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
param_dim = len(sc.ground_truth_par)
prior = dists.StructDist({'param': dists.MvNormal(loc=np.zeros(param_dim),
                                                  cov=sc.prior_marg_var*np.eye(param_dim))})
gt = sc.ground_truth_par
true_sol = ModelExact.exact_sol(gt).reshape(-1,1)  # fixed "truth" used by all trials

observed_data = true_sol + np.random.normal(loc=0., scale=np.sqrt(sc.data_noise_var), size=true_sol.shape)

chain_setup, chain_seq = get_chain_seq(prior,
                                       sc.data_noise_var,
                                       observed_data,
                                       sc.len_chain,
                                       sc.fine_over_coarse_nsamp)

# Storage (lists; will save as arrays)
avg_runtimes_ansamp = []
avg_sim_cost_ansamp = []
mse_values_ansamp = []
mse_stddev_ansamp = []
avg_last_nsamps_ansamp = []
post_mean_val_ansamp = []
post_mean_stddev_ansamp = []

@ray.remote
def wrapped_trial(*args):
    # Simply delegate to your existing runner
    return run_single_smc_trial(*args)


# -----------------------------
# Submit ALL tasks up-front
# -----------------------------
submitted_refs = []
ref_to_chain = {}

print("Submitting all Ray tasks...")
submit_start = time.time()

for chain in chain_seq:
    for run_idx in range(N_runs):
        ref = wrapped_trial.remote(
            sc.data_noise_var,
            true_sol,
            prior,
            gt,
            chain, #TODO
            sc.Nmin,
            sc.ar_tol,
            sc.ESS_rmin,
            sc.n_particles,
            resample_move_end,
            run_idx,
            observed_data
        )
        submitted_refs.append(ref)
        ref_to_chain[ref] = chain

print(f"Submitted {len(submitted_refs)} tasks in {time.time() - submit_start:.2f}s.")

# -----------------------------
# Collect results as they finish
# -----------------------------
results_by_chain = defaultdict(list)

pending = set(submitted_refs)
n_done = 0
total = len(submitted_refs)
t0 = time.time()

print("Collecting results as they finish...")
while pending:
    # Wait for next finished task
    ready, pending = ray.wait(list(pending), num_returns=1, timeout=None)
    ref = ready[0]
    chain = ref_to_chain[ref]

    # Get its result and store
    result = ray.get(ref)
    results_by_chain[chain].append(result)

    n_done += 1
    if n_done % max(1, total // 20) == 0 or n_done == total:
        elapsed = time.time() - t0
        print(f"[{n_done}/{total}] collected in {elapsed:.1f}s")

print(f"All results collected in {time.time() - t0:.2f}s.")

# -----------------------------
# Reduce to aggregates per noise level (in original order)
# -----------------------------
for chain in chain_seq:
    res_list = results_by_chain[chain]
    # Unpack tuples
    (
        runtimes_ansamp,
        mse_ansamp,
        sim_cost_ansamp,
        nsamps_last_ansamp,
        n_temp_steps_ansamp,
        post_means_ansamp
    ) = zip(*res_list)

    # Means
    post_mean_val_ansamp.append(np.mean(post_means_ansamp, axis=0))
    avg_runtimes_ansamp.append(np.mean(runtimes_ansamp))
    avg_sim_cost_ansamp.append(np.mean(sim_cost_ansamp))
    mse_values_ansamp.append(np.mean(mse_ansamp))
    post_mean_stddev_ansamp.append(np.std(post_means_ansamp,
                                          axis=0,
                                          ddof=1))
    mse_stddev_ansamp.append(np.std(mse_ansamp, ddof=1))
    avg_last_nsamps_ansamp.append(np.mean(nsamps_last_ansamp))

# -----------------------------
# Save everything in your format
# -----------------------------
np.savez(
    os.path.join(RESULTS_FOLDER, "smc_results.npz"),
    n_runs=N_runs,
    observed_data=observed_data, #Can be None
    avg_runtimes_ansamp=np.array(avg_runtimes_ansamp),
    avg_sim_cost_ansamp=np.array(avg_sim_cost_ansamp),
    mse_values_ansamp=np.array(mse_values_ansamp),
    mse_stddev_ansamp=np.array(mse_stddev_ansamp),
    avg_last_nsamps_ansamp=np.array(avg_last_nsamps_ansamp),
    post_mean_val_ansamp=np.array(post_mean_val_ansamp),
    post_mean_stddev_ansamp=np.array(post_mean_stddev_ansamp)
)

with open(os.path.join(RESULTS_FOLDER, "chain_setup.json"), "w") as f:
    json.dump(chain_setup, f, indent=4)

print(f"\nSaved results to {os.path.join(RESULTS_FOLDER, 'smc_results.npz')} and ...chain_setup.json")