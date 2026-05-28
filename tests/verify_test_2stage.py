import os
import shutil
import numpy as np
import ray
from scipy.stats import wasserstein_distance
from particles import distributions as dists
from particles import smc_samplers as ssp
from particles.core import SMC
from problem_setup.wos.problem_parameters import SamplerConfig as sc
from problem_setup.wos.likelihood_model import WOSHD as ModelExact
from utils.is_get_ref_sol import get_ref_sols
from utils.is_trial import run_single_is_trial

from manager.experiment import ExperimentManager
from experiments.two_stage_mcmc import Test2StageMCMCExperiment
from omegaconf import OmegaConf

# -----------------------------
# Cheap Configuration
# -----------------------------
N_runs = 2
budgets = np.array([10])
percents_on_coarse = np.array([0.0, 0.5, 1.0])
SEED = 1111
WOS_ref_sol = False

# Output folders
LEGACY_FOLDER = "data/smc_results_legacy_cheap"
NEW_FOLDER = "data/smc_results_new_cheap"

def run_legacy():
    print("\n--- RUNNING LEGACY PIPELINE ---")
    if os.path.exists(LEGACY_FOLDER):
        shutil.rmtree(LEGACY_FOLDER)
    os.makedirs(LEGACY_FOLDER, exist_ok=True)
    
    np.random.seed(SEED)
    
    # Ray init (ensure ray is clean and has 1 cpu)
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=1)
    
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

    print("Computing legacy reference solutions...")
    post_samp_ref_exact, post_samp_ref_wos = get_ref_sols(
        prior,
        sc.data_noise_var,
        observed_data,
        sc.Nmin,
        WOS_ref_sol
    )

    print("Computing legacy initial errors...")
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

    # Submit tasks
    @ray.remote
    def wrapped_trial(*args):
        return run_single_is_trial(*args)

    submitted_refs = []
    ref_to_setup = {}

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

    W_ex_by_setup = []
    W_wos_by_setup = []
    Wmax_ex_by_setup = []
    Wmax_wos_by_setup = []
    Ww_ex_by_setup = []
    Ww_wos_by_setup = []
    setups = []

    pending = set(submitted_refs)
    while pending:
        ready, pending = ray.wait(list(pending), num_returns=1)
        ref = ready[0]
        setup = ref_to_setup[ref]

        result = ray.get(ref)
        N, par_dim = result.shape

        W_ex = np.zeros(par_dim)
        W_wos = np.zeros(par_dim)

        for i in range(par_dim):
            W_ex[i] = wasserstein_distance(result[:, i], post_samp_ref_exact[:, i])
            W_wos[i] = wasserstein_distance(result[:, i], post_samp_ref_wos[:, i])

        W_ex_by_setup.append(np.mean(W_ex))
        W_wos_by_setup.append(np.mean(W_wos))
        Wmax_ex_by_setup.append(np.max(W_ex))
        Wmax_wos_by_setup.append(np.max(W_wos))
        Ww_ex_by_setup.append(
            np.average(W_ex, weights=1 / np.arange(1, par_dim + 1) ** ModelExact.q)
        )
        Ww_wos_by_setup.append(
            np.average(W_wos, weights=1 / np.arange(1, par_dim + 1) ** ModelExact.q)
        )
        setups.append(setup)

    np.savez(
        os.path.join(LEGACY_FOLDER, "smc_results.npz"),
        n_runs=N_runs,
        observed_data=observed_data,
        W_ex_by_setup=np.array(W_ex_by_setup),
        W_wos_by_setup=np.array(W_wos_by_setup),
        Wmax_ex_by_setup=np.array(Wmax_ex_by_setup),
        Wmax_wos_by_setup=np.array(Wmax_wos_by_setup),
        Ww_ex_by_setup=np.array(Ww_ex_by_setup),
        Ww_wos_by_setup=np.array(Ww_wos_by_setup),
        init_Wmean_ex=init_Wmean_ex,
        init_Wmax_ex=init_Wmax_ex,
        init_Ww_ex=init_Ww_ex,
        setups=np.array(setups),
        WOS_ref_sol=WOS_ref_sol
    )
    print("Legacy run saved.")
    ray.shutdown()


def run_new():
    print("\n--- RUNNING NEW OOP FRAMEWORK PIPELINE ---")
    if os.path.exists(NEW_FOLDER):
        shutil.rmtree(NEW_FOLDER)
    os.makedirs(NEW_FOLDER, exist_ok=True)
    
    np.random.seed(SEED)
    
    # Ray init clean
    if ray.is_initialized():
        ray.shutdown()
        
    cfg = OmegaConf.create({
        "num_cpus": 1,
        "results_base_dir": LEGACY_FOLDER, # We override base_dir inside runner or manager, wait, NEW_FOLDER!
        "WOS_ref_sol": False,
        "problem": {
            "name": "wos",
            "path": "problem_setup.wos",
            "model_exact_class": "WOSHD",
            "param_dim": sc.param_dim,
            "data_noise_var": sc.data_noise_var,
            "prior_marg_var": sc.prior_marg_var,
            "prior_type": "mvnormal",
            "n_particles": sc.n_particles,
            "Nmin": sc.Nmin,
            "fine_over_coarse_nsamp": sc.fine_over_coarse_nsamp,
            "likelihood_suffix": "HD",
            "ground_truth_par": sc.ground_truth_par.tolist()
        },
        "sweep": {
            "n_runs": N_runs,
            "budgets": budgets.tolist(),
            "percents": percents_on_coarse.tolist()
        }
    })
    
    # To save results in NEW_FOLDER directly:
    # ExperimentManager uses info.create_results_folder(base_dir=cfg.results_base_dir)
    # We will pass base_dir=NEW_FOLDER
    cfg.results_base_dir = NEW_FOLDER
    
    runner = Test2StageMCMCExperiment()
    manager = ExperimentManager(runner, cfg)
    
    output_dir = manager.run()
    
    # Rename smc_results.npz to NEW_FOLDER/smc_results.npz so it is in a standard place
    src_file = os.path.join(output_dir, "smc_results.npz")
    dst_file = os.path.join(NEW_FOLDER, "smc_results.npz")
    shutil.copy2(src_file, dst_file)
    print(f"New OOP framework results saved and copied to {dst_file}")
    ray.shutdown()


def compare_and_assert():
    legacy_file = os.path.join(LEGACY_FOLDER, "smc_results.npz")
    new_file = os.path.join(NEW_FOLDER, "smc_results.npz")
    
    legacy_data = np.load(legacy_file, allow_pickle=True)
    new_data = np.load(new_file, allow_pickle=True)
    
    # 1. Compare non-asynchronous parts (observed data, initial errors)
    assert np.allclose(legacy_data["observed_data"], new_data["observed_data"], atol=1e-5), "Observed data mismatch!"
    assert np.allclose(legacy_data["init_Wmean_ex"], new_data["init_Wmean_ex"], atol=1e-5), "init_Wmean_ex mismatch!"
    assert np.allclose(legacy_data["init_Wmax_ex"], new_data["init_Wmax_ex"], atol=1e-5), "init_Wmax_ex mismatch!"
    assert np.allclose(legacy_data["init_Ww_ex"], new_data["init_Ww_ex"], atol=1e-5), "init_Ww_ex mismatch!"
    
    # 2. Sort completed tasks by budget & percent to ensure order-invariant comparison
    leg_setups = legacy_data["setups"]
    new_setups = new_data["setups"]
    
    leg_stats = {
        "W_ex": legacy_data["W_ex_by_setup"],
        "W_wos": legacy_data["W_wos_by_setup"],
        "Wmax_ex": legacy_data["Wmax_ex_by_setup"],
        "Wmax_wos": legacy_data["Wmax_wos_by_setup"],
        "Ww_ex": legacy_data["Ww_ex_by_setup"],
        "Ww_wos": legacy_data["Ww_wos_by_setup"]
    }
    new_stats = {
        "W_ex": new_data["W_ex_by_setup"],
        "W_wos": new_data["W_wos_by_setup"],
        "Wmax_ex": new_data["Wmax_ex_by_setup"],
        "Wmax_wos": new_data["Wmax_wos_by_setup"],
        "Ww_ex": new_data["Ww_ex_by_setup"],
        "Ww_wos": new_data["Ww_wos_by_setup"]
    }
    
    # Sort legacy results by setups (budget, percent)
    leg_indices = sorted(range(len(leg_setups)), key=lambda k: (leg_setups[k][0], leg_setups[k][1]))
    new_indices = sorted(range(len(new_setups)), key=lambda k: (new_setups[k][0], new_setups[k][1]))
    
    # Assert sorted setups match
    sorted_leg_setups = leg_setups[leg_indices]
    sorted_new_setups = new_setups[new_indices]
    assert np.allclose(sorted_leg_setups, sorted_new_setups, atol=1e-5), "Setups mismatch!"
    
    print("\n--- COMPARING SMC RESULTS (TOLERANCE 1e-4) ---")
    for key in leg_stats.keys():
        sorted_leg_stat = leg_stats[key][leg_indices]
        sorted_new_stat = new_stats[key][new_indices]
        
        max_diff = np.max(np.abs(sorted_leg_stat - sorted_new_stat))
        mean_diff = np.mean(np.abs(sorted_leg_stat - sorted_new_stat))
        print(f"{key:10s} - Max Diff: {max_diff:.2e}, Mean Diff: {mean_diff:.2e}")
        
        assert max_diff <= 1e-4, f"Mismatch in {key} exceeds 1e-4 tolerance!"
        
    print("\nSUCCESS: The new dynamic OOP pipeline results match the legacy pipeline perfectly!")


if __name__ == "__main__":
    run_legacy()
    run_new()
    compare_and_assert()
