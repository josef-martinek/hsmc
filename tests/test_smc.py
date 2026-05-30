import os
import numpy as np
import pytest
from particles import distributions as dists
from particles import smc_samplers as ssp
from particles.core import SMC
import importlib

# Ensure reproducible results
def set_seeds():
    np.random.seed(1111)

def run_smc_problem(problem_name, model_exact_class, param_dim, prior_marg_var, data_noise_var, n_particles, len_chain, ESS_rmin, class_path, ground_truth_par, Nmin, ar_tol):
    set_seeds()
    
    # 1. Dynamic imports of likelihood model classes
    prob_lik = importlib.import_module(f"{class_path}.likelihood_model")
    ModelExact = getattr(prob_lik, model_exact_class)
    IPModelInexactAdaptHD = getattr(prob_lik, "IPModelInexactAdaptHD")
    
    # 2. Parse ground truth parameter
    ground_truth_par = np.asarray(ground_truth_par)
    
    prior = dists.StructDist({'param': dists.MvNormal(loc=np.zeros(param_dim),
                                                      cov=prior_marg_var * np.eye(param_dim))})
    
    # Generate synthetic data
    true_sol = ModelExact.exact_sol(ground_truth_par).reshape(-1, 1)
    rng_data = np.random.default_rng(seed=81)
    observed_data = true_sol + rng_data.normal(loc=0., scale=np.sqrt(data_noise_var), size=true_sol.shape)
    
    model = IPModelInexactAdaptHD(prior=prior, data_noise_cov=data_noise_var, data=observed_data)
    
    # Simple Random Walk Adaptive Tempering using TempAdaptNsampNoRew
    import core.mcmc as MCMC
    from core.samplers import TempAdaptNsampNoRew
    mcmc = MCMC.RWRegCal()
    move = ssp.AdaptiveMCMCSequence(mcmc=mcmc, adaptive=False, len_chain=len_chain)
    
    fk_tempering = TempAdaptNsampNoRew(
        Nmin=Nmin,
        ar_tol=ar_tol,
        model=model,
        wastefree=False,
        ESSrmin=ESS_rmin,
        move=move,
        len_chain=len_chain
    )
    
    my_temp_alg = SMC(fk=fk_tempering, N=n_particles, ESSrmin=1., verbose=False)
    my_temp_alg.run()
    my_temp_alg.setup_auxiliary_weights()
    my_temp_alg.resample_move()
    
    post_samp = my_temp_alg.X.theta['param']
    post_mean = np.mean(post_samp, axis=0)
    
    return post_mean, post_samp

def check_ref(problem_name, post_mean, post_samp, hydra_tag=""):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    ref_file = os.path.join(data_dir, f"{problem_name}_ref.npz")
    
    # Read-only verification: must read from existing data
    ref = np.load(ref_file)
    
    diff_mean_max = np.max(np.abs(post_mean - ref['post_mean']))
    diff_mean_avg = np.mean(np.abs(post_mean - ref['post_mean']))
    diff_samp_max = np.max(np.abs(post_samp - ref['post_samp']))
    diff_samp_avg = np.mean(np.abs(post_samp - ref['post_samp']))
    
    tag = f" ({hydra_tag})" if hydra_tag else ""
    print(f"\n[{problem_name.upper()} SMC DIFFERENCES AGAINST REFERENCE{tag}]")
    print(f"  - post_mean: max_abs_diff = {diff_mean_max:.2e}, mean_abs_diff = {diff_mean_avg:.2e}")
    print(f"  - post_samp: max_abs_diff = {diff_samp_max:.2e}, mean_abs_diff = {diff_samp_avg:.2e}")
    
    assert np.allclose(post_mean, ref['post_mean'], atol=1e-5), f"{problem_name} post_mean mismatch!"
    assert np.allclose(post_samp, ref['post_samp'], atol=1e-5), f"{problem_name} post_samp mismatch!"

def test_wos_smc():
    post_mean, post_samp = run_smc_problem(
        problem_name="wos",
        model_exact_class="WOSHD",
        param_dim=2,
        prior_marg_var=4.0,
        data_noise_var=1.6,
        n_particles=20,
        len_chain=5,
        ESS_rmin=0.7,
        class_path="problem_setup.wos",
        ground_truth_par=[-3.2077257, -5.29743598],
        Nmin=32,
        ar_tol=0.2
    )
    check_ref("wos", post_mean, post_samp)

def test_toy_smc():
    post_mean, post_samp = run_smc_problem(
        problem_name="toy",
        model_exact_class="MatVecModel",
        param_dim=24,
        prior_marg_var=1.0,
        data_noise_var=0.1,
        n_particles=20,
        len_chain=5,
        ESS_rmin=0.7,
        class_path="problem_setup.toy_problem",
        ground_truth_par=np.ones((24, 1)),
        Nmin=100,
        ar_tol=0.35
    )
    check_ref("toy", post_mean, post_samp)

