import os
import time
from collections import defaultdict
import numpy as np
import ray
from particles import distributions as dists
from problem_setup.wos.problem_parameters import SamplerConfig as sc
from problem_setup.wos.likelihood_model import WOSHD as ModelExact
from utils.is_get_ref_sol import get_ref_sols
from scipy.stats import wasserstein_distance
import utils.export_info as info
import utils
import json

if sc.param_dim == 1:
    raise NotImplementedError
else:
    from utils.is_trial import run_single_is_trial

# -----------------------------
# Config
# -----------------------------
RESULTS_FOLDER = "data/smc_results_test"   # or info.create_results_folder()
#RESULTS_FOLDER = info.create_results_folder()
NUM_CPUS = 8
SEED = 1111
resample_move_end = True
WOS_ref_sol = False # If this is false, then WOS reference solution contains zeros.

# Each dim computed separately for now
# Nr of inference runs for each param in param_seq
N_runs = 10
percent_step = 2
percent_values = int(100/percent_step)+1

percents_on_coarse = np.linspace(0, 1., percent_values)
budgets = np.array([10, 20, 40, 80]) #in terms of fine chain STEPS


# -----------------------------
# Housekeeping / provenance
# -----------------------------
os.makedirs(RESULTS_FOLDER, exist_ok=True)
src = __file__
np.random.seed(SEED)

# Export reproducibility info
info.export_full_conda_env(target_folder=RESULTS_FOLDER)
info.copy_problem_setup(target_folder=RESULTS_FOLDER, module=sc)
info.copy_problem_setup(target_folder=RESULTS_FOLDER, module=utils.is_get_ref_sol)
info.copy_problem_setup(target_folder=RESULTS_FOLDER, module=utils.is_trial)
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
param_dim = sc.param_dim
prior = dists.StructDist({'param': dists.MvNormal(loc=np.zeros(param_dim),
                                                  cov=sc.prior_marg_var*np.eye(param_dim))})
gt = sc.ground_truth_par
true_sol = ModelExact.exact_sol(gt).reshape(-1,1)  # fixed "truth" used by all trials

rng_data = np.random.default_rng(seed=81)
observed_data = true_sol + rng_data.normal(loc=0., scale=np.sqrt(sc.data_noise_var), size=true_sol.shape)

print("Computing reference solutions...")
post_samp_ref_exact, post_samp_ref_wos = get_ref_sols(prior,
                                                      sc.data_noise_var,
                                                      observed_data,
                                                      sc.Nmin,
                                                      WOS_ref_sol)
print("Reference solutions computed.")

# Storage (lists; will save as arrays)
avg_runtimes = []

@ray.remote
def wrapped_trial(*args):
    # Simply delegate to your existing runner
    return run_single_is_trial(*args)


# -----------------------------
# Submit ALL tasks up-front
# -----------------------------
submitted_refs = []
ref_to_setup = {}

print("Submitting all Ray tasks...")
submit_start = time.time()

for budget in budgets:
    for percent in percents_on_coarse:
        for run_idx in range(N_runs):
            ref = wrapped_trial.remote(
                sc.data_noise_var,
                prior,
                sc.Nmin,
                sc.n_particles,
                sc.fine_over_coarse_nsamp,
                run_idx,
                observed_data,
                budget,
                percent
            )
            submitted_refs.append(ref)
            ref_to_setup[ref] = [budget, percent]

print(f"Submitted {len(submitted_refs)} tasks in {time.time() - submit_start:.2f}s.")

# -----------------------------
# Collect results as they finish
# -----------------------------
W_ex_by_setup = []
W_wos_by_setup = []
Wmax_ex_by_setup = []
Wmax_wos_by_setup = []
setups = []

pending = set(submitted_refs)
n_done = 0
total = len(submitted_refs)
t0 = time.time()

print("Collecting results as they finish...")
while pending:
    # Wait for next finished task
    ready, pending = ray.wait(list(pending), num_returns=1, timeout=None)
    ref = ready[0]
    setup = ref_to_setup[ref]

    # Get its result and store
    result = ray.get(ref) # result is posterior samples
    N, par_dim = result.shape
    W_wrt_ex = np.zeros(par_dim)
    W_wrt_wos = np.zeros(par_dim)
    for i in range(par_dim):
        W_wrt_ex[i] = wasserstein_distance(result[:, i],
                                           post_samp_ref_exact[:,i])
        W_wrt_wos[i] = wasserstein_distance(result[:, i],
                                            post_samp_ref_wos[:,i])
    Wmax_wrt_ex = np.max(W_wrt_ex)
    Wmax_wrt_wos = np.max(W_wrt_wos)
    Wmean_wrt_ex = np.mean(W_wrt_ex)
    Wmean_wrt_wos = np.mean(W_wrt_wos)
    W_ex_by_setup.append(Wmean_wrt_ex)
    W_wos_by_setup.append(Wmean_wrt_wos)
    Wmax_ex_by_setup.append(Wmax_wrt_ex)
    Wmax_wos_by_setup.append(Wmax_wos_by_setup)
    setups.append(setup)
    #print("Setup:", setup)
    #print("Wass dist:", Wmean_wrt_ex)
    #print("")

    n_done += 1
    if n_done % max(1, total // 20) == 0 or n_done == total:
        elapsed = time.time() - t0
        print(f"[{n_done}/{total}] collected in {elapsed:.1f}s")

print(f"All results collected in {time.time() - t0:.2f}s.")

# -----------------------------
# Save everything in your format
# -----------------------------
np.savez(
    os.path.join(RESULTS_FOLDER, "smc_results.npz"),
    n_runs=N_runs,
    observed_data=observed_data, #Can be None
    avg_runtimes_ansamp=np.array(avg_runtimes), #TODO this is empty
    W_ex_by_setup=np.array(W_ex_by_setup),
    W_wos_by_setup=np.array(W_wos_by_setup),
    Wmax_wos_by_setup=np.array(Wmax_wos_by_setup),
    Wmax_ex_by_setup=np.array(Wmax_ex_by_setup),
    setups=np.array(setups),
    WOS_ref_sol = WOS_ref_sol
)

print(f"\nSaved results to {os.path.join(RESULTS_FOLDER, 'smc_results.npz')} and ...chain_setup.json")