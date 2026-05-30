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
RESULTS_FOLDER = "data/smc_results_test"
#RESULTS_FOLDER = info.create_results_folder()
NUM_CPUS = 8
SEED = 1111
resample_move_end = True


N_runs = 10
percent_step = 2
percent_values = int(100 / percent_step) + 1

percents_on_coarse = np.linspace(0, 1., percent_values)
budgets = np.array([10, 20, 40, 80]) #in terms of fine chain STEPS

# -----------------------------
# Housekeeping / provenance
# -----------------------------
os.makedirs(RESULTS_FOLDER, exist_ok=True)
src = __file__
np.random.seed(SEED)

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
    ray.init(ignore_reinit_error=True)

# -----------------------------
# Problem setup
# -----------------------------
param_dim = sc.param_dim
prior = dists.StructDist({
    'param': dists.MvNormal(
        loc=np.zeros(param_dim),
        cov=sc.prior_marg_var * np.eye(param_dim)
    )
})
gt = sc.ground_truth_par
true_sol = ModelExact.exact_sol(gt).reshape(-1, 1)

rng_data = np.random.default_rng(seed=81)
observed_data = true_sol + rng_data.normal(
    loc=0.,
    scale=np.sqrt(sc.data_noise_var),
    size=true_sol.shape
)

print("Computing reference solutions...")
post_samp_ref_exact = get_ref_sols(
    prior,
    sc.data_noise_var,
    observed_data
)
print("Reference solutions computed.")

# -----------------------------
# INITIAL ERROR (NEW)
# -----------------------------
print("Computing initial (prior) errors...")
N_init = 100
W_init_ex = []
Wmax_init_ex = []
Ww_init_ex = []

for _ in range(N_init):
    prior_samples = prior.rvs(sc.n_particles)['param']
    W = np.zeros(param_dim)
    for i in range(param_dim):
        W[i] = wasserstein_distance(
            prior_samples[:, i],
            post_samp_ref_exact[:, i]
        )
    W_init_ex.append(np.mean(W))
    Wmax_init_ex.append(np.max(W))
    Ww_init_ex.append(
        np.average(W, weights=1 / np.arange(1, param_dim + 1) ** ModelExact.q)
    )

init_Wmean_ex = np.mean(W_init_ex)
init_Wmax_ex = np.mean(Wmax_init_ex)
init_Ww_ex = np.mean(Ww_init_ex)

print("Initial errors computed.")

# -----------------------------
# Storage
# -----------------------------
avg_runtimes = []

@ray.remote
def wrapped_trial(*args):
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
# Collect results
# -----------------------------
W_ex_by_setup = []
Wmax_ex_by_setup = []
Ww_ex_by_setup = []
setups = []

pending = set(submitted_refs)
n_done = 0
total = len(submitted_refs)
t0 = time.time()

print("Collecting results as they finish...")
while pending:
    ready, pending = ray.wait(list(pending), num_returns=1)
    ref = ready[0]
    setup = ref_to_setup[ref]

    result = ray.get(ref)
    N, par_dim = result.shape

    W_ex = np.zeros(par_dim)

    for i in range(par_dim):
        W_ex[i] = wasserstein_distance(result[:, i],
                                       post_samp_ref_exact[:, i])

    W_ex_by_setup.append(np.mean(W_ex))
    Wmax_ex_by_setup.append(np.max(W_ex))
    Ww_ex_by_setup.append(
        np.average(W_ex, weights=1 / np.arange(1, par_dim + 1) ** ModelExact.q)
    )
    setups.append(setup)

    n_done += 1
    if n_done % max(1, total // 20) == 0 or n_done == total:
        print(f"[{n_done}/{total}] collected in {time.time() - t0:.1f}s")

print(f"All results collected in {time.time() - t0:.2f}s.")

# -----------------------------
# Save
# -----------------------------
np.savez(
    os.path.join(RESULTS_FOLDER, "smc_results.npz"),
    n_runs=N_runs,
    observed_data=observed_data,
    avg_runtimes_ansamp=np.array(avg_runtimes),
    W_ex_by_setup=np.array(W_ex_by_setup),
    Wmax_ex_by_setup=np.array(Wmax_ex_by_setup),
    Ww_ex_by_setup=np.array(Ww_ex_by_setup),
    init_Wmean_ex=init_Wmean_ex,
    init_Wmax_ex=init_Wmax_ex,
    init_Ww_ex=init_Ww_ex,
    setups=np.array(setups)
)

print(f"\nSaved results to {os.path.join(RESULTS_FOLDER, 'smc_results.npz')}")
