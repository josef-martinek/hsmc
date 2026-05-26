import os
import numpy as np
import pytest
from particles import distributions as dists
from particles.smc_samplers import MCMCSequenceWF, ThetaParticles, Tempering
from core.mcmc import RWUnc, NoisyMetropolisUnc, RWRegCal, NoisyMetropolisCal, TwoStageMCMC
import importlib

def run_mcmc_test():
    np.random.seed(1111)
    
    prob_lik = importlib.import_module("problem_setup.toy_problem.likelihood_model")
    ModelExact = prob_lik.MatVecModel
    IPModelInexactFixHD = prob_lik.IPModelInexactFixHD
    IPModelInexactAdaptHD = prob_lik.IPModelInexactAdaptHD
    
    # Hardcoded config parameters matching legacy toy problem setup
    param_dim = 24
    prior_marg_var = 1.0
    data_noise_var = 0.1
    ground_truth_par = np.ones((param_dim, 1))
    
    prior = dists.StructDist({'param': dists.MvNormal(loc=np.zeros(param_dim),
                                                      cov=prior_marg_var * np.eye(param_dim))})
    
    # Generate synthetic data
    true_sol = ModelExact.exact_sol(ground_truth_par).reshape(-1, 1)
    rng_data = np.random.default_rng(seed=81)
    observed_data = true_sol + rng_data.normal(loc=0., scale=np.sqrt(data_noise_var), size=true_sol.shape)
    
    model_fine = IPModelInexactFixHD(prior=prior, data_noise_cov=data_noise_var, data=observed_data, nsamp=100)
    model_coarse = IPModelInexactAdaptHD(prior=prior, data_noise_cov=data_noise_var, data=observed_data)
    
    x0 = ThetaParticles(theta=prior.rvs(size=1))
    
    # Initialize necessary MCMC sequence shared variables to avoid KeyError
    x0.shared["solver_nsamp"] = [100]
    x0.shared["exponents"] = [1.0]
    
    aux_temp_alg_fine = Tempering(model=model_fine, exponents=np.array([0., 1.]))
    target_fine = Tempering.current_target(aux_temp_alg_fine, 1.)
    target_fine(x0)
    
    # 1. RWUnc
    pmmcmc = MCMCSequenceWF(mcmc=RWUnc(scale_factor=0.2), len_chain=10)
    samp_pm = pmmcmc(x=x0.copy(), target=target_fine)
    
    # 2. NoisyMetropolisUnc
    mcwm = MCMCSequenceWF(mcmc=NoisyMetropolisUnc(scale_factor=1.0), len_chain=10)
    samp_mcwm = mcwm(x=x0.copy(), target=target_fine)
    
    # 3. RWRegCal (use eye_prop=True to bypass 0-variance cholesky on single particle)
    reg_cal = MCMCSequenceWF(mcmc=RWRegCal(eye_prop=True), len_chain=10)
    reg_cal.mcmc.calibrate(W=np.array([1.0]), x=x0)
    samp_reg = reg_cal(x=x0.copy(), target=target_fine)
    
    # 4. NoisyMetropolisCal (provide manual chol_cov and skip dynamic cholesky calibration)
    nm_cal = MCMCSequenceWF(mcmc=NoisyMetropolisCal(scale=1.0), len_chain=10)
    x0.shared["chol_cov"] = np.eye(param_dim)
    samp_nm_cal = nm_cal(x=x0.copy(), target=target_fine)
    
    # 5. TwoStageMCMC
    two_stage = TwoStageMCMC(
        mcmc1=RWUnc(scale_factor=0.2),
        len_chain1=3,
        mcmc2=RWUnc(scale_factor=0.2),
        len_chain2=3,
        coarse_model=model_coarse,
        fine_over_coarse=10
    )
    two_stage.calibrate(W=np.array([1.0]), x=x0)
    samp_two_stage = two_stage(x=x0.copy(), target_fine=target_fine)
    
    return (
        samp_pm.theta['param'], 
        samp_mcwm.theta['param'], 
        samp_reg.theta['param'], 
        samp_nm_cal.theta['param'], 
        samp_two_stage.theta['param']
    )

def run_mcmc_test_hydra():
    np.random.seed(1111)
    
    from hydra import initialize, compose
    from hydra.utils import instantiate
    
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["problem=toy"])
    
    prob_lik = importlib.import_module(f"{cfg.problem.path}.likelihood_model")
    ModelExact = getattr(prob_lik, cfg.problem.model_exact_class)
    
    param_dim = cfg.problem.param_dim
    ground_truth_par = np.ones((param_dim, 1))
    
    prior = dists.StructDist({'param': dists.MvNormal(loc=np.zeros(param_dim),
                                                      cov=cfg.problem.prior_marg_var * np.eye(param_dim))})
    
    true_sol = ModelExact.exact_sol(ground_truth_par).reshape(-1, 1)
    rng_data = np.random.default_rng(seed=81)
    observed_data = true_sol + rng_data.normal(loc=0., scale=np.sqrt(cfg.problem.data_noise_var), size=true_sol.shape)
    
    # 1. Instantiate Models via Hydra
    with initialize(version_base=None, config_path="../conf"):
        cfg_fine = compose(config_name="config", overrides=["problem=toy", "likelihood=inexact_fix", "likelihood.nsamp=100"])
        cfg_coarse = compose(config_name="config", overrides=["problem=toy", "likelihood=inexact_adapt"])
        
    model_fine = instantiate(cfg_fine.likelihood, prior=prior, data_noise_cov=cfg.problem.data_noise_var, data=observed_data)
    model_coarse = instantiate(cfg_coarse.likelihood, prior=prior, data_noise_cov=cfg.problem.data_noise_var, data=observed_data)
    
    x0 = ThetaParticles(theta=prior.rvs(size=1))
    x0.shared["solver_nsamp"] = [100]
    x0.shared["exponents"] = [1.0]
    
    aux_temp_alg_fine = Tempering(model=model_fine, exponents=np.array([0., 1.]))
    target_fine = Tempering.current_target(aux_temp_alg_fine, 1.)
    target_fine(x0)
    
    # 2. Instantiate MCMC mutations via Hydra
    with initialize(version_base=None, config_path="../conf"):
        cfg_rw_unc = compose(config_name="config", overrides=["mcmc=rw_unc", "mcmc.scale_factor=0.2"])
        cfg_nm_unc = compose(config_name="config", overrides=["mcmc=noisy_metropolis_unc", "mcmc.scale_factor=1.0"])
        cfg_rw_reg = compose(config_name="config", overrides=["mcmc=rw_reg_cal"])
        cfg_nm_cal = compose(config_name="config", overrides=["mcmc=noisy_metropolis_cal", "mcmc.scale=1.0"])
        cfg_two_stage = compose(config_name="config", overrides=["mcmc=two_stage"])
        
    mcmc_pm = instantiate(cfg_rw_unc.mcmc)
    mcmc_mcwm = instantiate(cfg_nm_unc.mcmc)
    mcmc_reg = instantiate(cfg_rw_reg.mcmc, eye_prop=True)
    mcmc_nm_cal = instantiate(cfg_nm_cal.mcmc)
    mcmc_two_stage = instantiate(cfg_two_stage.mcmc, coarse_model=model_coarse)
    
    # Run MCMC sequences
    pmmcmc = MCMCSequenceWF(mcmc=mcmc_pm, len_chain=10)
    samp_pm = pmmcmc(x=x0.copy(), target=target_fine)
    
    mcwm = MCMCSequenceWF(mcmc=mcmc_mcwm, len_chain=10)
    samp_mcwm = mcwm(x=x0.copy(), target=target_fine)
    
    reg_cal = MCMCSequenceWF(mcmc=mcmc_reg, len_chain=10)
    reg_cal.mcmc.calibrate(W=np.array([1.0]), x=x0)
    samp_reg = reg_cal(x=x0.copy(), target=target_fine)
    
    nm_cal = MCMCSequenceWF(mcmc=mcmc_nm_cal, len_chain=10)
    x0.shared["chol_cov"] = np.eye(param_dim)
    samp_nm_cal = nm_cal(x=x0.copy(), target=target_fine)
    
    mcmc_two_stage.calibrate(W=np.array([1.0]), x=x0)
    samp_two_stage = mcmc_two_stage(x=x0.copy(), target_fine=target_fine)
    
    return (
        samp_pm.theta['param'], 
        samp_mcwm.theta['param'], 
        samp_reg.theta['param'], 
        samp_nm_cal.theta['param'], 
        samp_two_stage.theta['param']
    )

def test_mcmc_correctness():
    pm, mcwm, reg, nm_cal, two_stage = run_mcmc_test()
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    ref_file = os.path.join(data_dir, "mcmc_ref.npz")
    
    # Read-only verification: must read from existing data
    ref = np.load(ref_file)
    
    print("\n[MCMC OPERATORS DIFFERENCES AGAINST REFERENCE]")
    methods = [
        ("PM", pm, ref['samp_pm']),
        ("MCWM", mcwm, ref['samp_mcwm']),
        ("Reg", reg, ref['samp_reg']),
        ("NM Cal", nm_cal, ref['samp_nm_cal']),
        ("TwoStage", two_stage, ref['samp_two_stage'])
    ]
    for name, new, reference in methods:
        diff_max = np.max(np.abs(new - reference))
        diff_avg = np.mean(np.abs(new - reference))
        print(f"  - {name:8s}: max_abs_diff = {diff_max:.2e}, mean_abs_diff = {diff_avg:.2e}")
        assert np.allclose(new, reference, atol=1e-5), f"MCMC {name} sample mismatch!"

def test_mcmc_correctness_hydra():
    pm, mcwm, reg, nm_cal, two_stage = run_mcmc_test_hydra()
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    ref_file = os.path.join(data_dir, "mcmc_ref.npz")
    
    ref = np.load(ref_file)
    
    print("\n[HYDRA MCMC OPERATORS DIFFERENCES AGAINST REFERENCE]")
    methods = [
        ("PM (Hydra)", pm, ref['samp_pm']),
        ("MCWM (Hydra)", mcwm, ref['samp_mcwm']),
        ("Reg (Hydra)", reg, ref['samp_reg']),
        ("NM Cal (Hydra)", nm_cal, ref['samp_nm_cal']),
        ("TwoStage (Hydra)", two_stage, ref['samp_two_stage'])
    ]
    for name, new, reference in methods:
        diff_max = np.max(np.abs(new - reference))
        diff_avg = np.mean(np.abs(new - reference))
        print(f"  - {name:18s}: max_abs_diff = {diff_max:.2e}, mean_abs_diff = {diff_avg:.2e}")
        assert np.allclose(new, reference, atol=1e-5), f"MCMC {name} sample mismatch!"
