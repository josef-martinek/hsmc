import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib
import numpy as np
import time
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from particles import distributions as dists
from particles.smc_samplers import AdaptiveMCMCSequence
from particles.core import SMC

resample_move_end = True

def run_simulation(problem="wos", likelihood="exact", mcmc="rw_reg_cal", sampler="adaptive_tempering", ground_truth_par=None, **sampler_overrides):
    """
    Run a single SMC tempering simulation using modular Hydra configurations.
    
    Parameters:
    -----------
    problem : str
        The problem config profile, e.g. "wos" or "toy".
    likelihood : str
        The likelihood profile, e.g. "exact", "inexact_fix", or "inexact_adapt".
    mcmc : str
        The MCMC step mutation profile, e.g. "rw_reg_cal", "rw_unc", or "noisy_metropolis_cal".
    sampler : str
        The SMC sampler profile, e.g. "adaptive_tempering", "temp_adapt_nsamp_no_rew", or "tempering".
    ground_truth_par : array-like, optional
        A custom parameter array to override the default ground truth loaded from YAML.
    **sampler_overrides : dict
        Dynamic keyword arguments to override properties of the sampler (e.g. exponents=[...]).
    """
    # ---------------------------------------------------------
    # 0. COMPOSE CONFIGURATION (Programmatic Compose API)
    # ---------------------------------------------------------
    overrides = [
        f"problem={problem}",
        f"likelihood={likelihood}",
        f"mcmc={mcmc}",
        f"sampler={sampler}"
    ]
    
    for key, val in sampler_overrides.items():
        if isinstance(val, list):
            val_str = "[" + ",".join(map(str, val)) + "]"
        else:
            val_str = str(val)
        overrides.append(f"sampler.{key}={val_str}")

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=overrides)

    # ---------------------------------------------------------
    # 1. DYNAMIC PROBLEM & LIKELIHOOD SETUP
    # ---------------------------------------------------------
    prob_lik = importlib.import_module(f"{cfg.problem.path}.likelihood_model")
    ModelExact = getattr(prob_lik, cfg.problem.model_exact_class)

    # Load default ground truth from YAML, or override with runtime argument
    if ground_truth_par is None:
        ground_truth_par = np.array(cfg.problem.ground_truth_par)
    else:
        ground_truth_par = np.asarray(ground_truth_par)

    param_dim = cfg.problem.param_dim
    if hasattr(cfg.problem, 'prior_type') and cfg.problem.prior_type == 'uniform':
        a = cfg.problem.prior_a
        b = cfg.problem.prior_b
        if param_dim == 1:
            prior = dists.StructDist({'param': dists.Uniform(a=a, b=b)})
        else:
            prior = dists.StructDist({'param': dists.IndepProd(*(dists.Uniform(a=a, b=b) for _ in range(param_dim)))})
    else:
        if param_dim == 1:
            prior = dists.StructDist({'param': dists.Normal(loc=0., 
                                                            scale=cfg.problem.prior_marg_var)})
        else:
            prior = dists.StructDist({'param': dists.MvNormal(loc=np.zeros(param_dim),
                                                              cov=cfg.problem.prior_marg_var*np.eye(param_dim))})

    # Generate synthetic data using ground truth exact solver
    true_sol = ModelExact.exact_sol(ground_truth_par).reshape(-1, 1)
    rng_data = np.random.default_rng(seed=81)
    observed_data = true_sol + rng_data.normal(loc=0., scale=np.sqrt(cfg.problem.data_noise_var), size=true_sol.shape)
    print(f"Observed data generated for problem '{problem}' using exact solver.")

    # Dynamic instantiation of the likelihood model wrapper
    model = instantiate(
        cfg.likelihood, 
        prior=prior, 
        data_noise_cov=cfg.problem.data_noise_var, 
        data=observed_data
    )

    # ---------------------------------------------------------
    # 2. MCMC & SAMPLER INSTANTIATION
    # ---------------------------------------------------------
    mcmc_obj = instantiate(cfg.mcmc)
    move = AdaptiveMCMCSequence(mcmc=mcmc_obj, adaptive=False, len_chain=cfg.problem.len_chain)
    fk_tempering = instantiate(cfg.sampler, model=model, move=move, len_chain=cfg.problem.len_chain)

    # ---------------------------------------------------------
    # 3. RUN SIMULATION
    # ---------------------------------------------------------
    my_temp_alg = SMC(fk=fk_tempering, N=cfg.problem.n_particles, ESSrmin=1., verbose=True)
    
    start_time = time.time()
    my_temp_alg.run()
    if resample_move_end:
        my_temp_alg.setup_auxiliary_weights()
        my_temp_alg.resample_move()
    end_time = time.time()
    
    print(f"Runtime: {end_time-start_time:.4f} seconds")

    post_samp = my_temp_alg.X.theta['param']
    post_mean = np.mean(post_samp, axis=0)
    print("Posterior mean:", post_mean)
    
    # Save output data
    os.makedirs("data/experiment", exist_ok=True)
    np.savez("data/experiment/post_mean_ref.npz", post_mean=post_mean)
    np.savez("data/experiment/post_samp_ref.npz", post_samp=post_samp)
    
    # Export execution setup configuration as reference
    with open("data/experiment/setup.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Plot results if 1D problem
    if param_dim == 1:
        import scipy.stats as stats
        from matplotlib import pyplot as plt
        import seaborn as sb

        # Define prior distribution
        prior_dist = stats.norm(loc=0, scale=cfg.problem.prior_marg_var)

        # Plot results
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot histogram of sampled parameters
        if resample_move_end:
            sb.histplot(x=post_samp.flatten(), stat='density', label='SMC Samples', ax=ax)
        else:
            sb.histplot(x=post_samp.flatten(), weights=my_temp_alg.W, stat='density', label='SMC Samples', ax=ax)

        # Plot prior distribution
        x_vals = np.linspace(-3 * cfg.problem.prior_marg_var, 3 * cfg.problem.prior_marg_var, 1000)
        ax.plot(x_vals, prior_dist.pdf(x_vals), label='Prior', linestyle='dashed', color='black')

        ax.set_xlabel('param')
        ax.legend()
        
        # Save the plot
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/posterior_distribution.png", dpi=300, bbox_inches='tight')
        print("Plot saved to plots/posterior_distribution.png")
        plt.show()

if __name__ == "__main__":
    # ---------------------------------------------------------
    # CONFIGURATION INTERFACE (Specify your run setup here)
    # ---------------------------------------------------------
    problem_name = "compact"        # Options: "wos", "toy", "compact"
    sampler = "adaptive_tempering"  # Options: "adaptive_tempering", "temp_adapt_nsamp_no_rew", "tempering"
    mcmc = "rw_reg_cal"             # Options: "rw_reg_cal", "rw_unc", "noisy_metropolis_cal"
    likelihood = "exact"            # Options: "exact", "inexact_fix", "inexact_adapt"

    print("==================================================")
    print("Running SMC Simulation")
    print(f"  Problem:    {problem_name}")
    print(f"  Sampler:    {sampler}")
    print(f"  MCMC Move:  {mcmc}")
    print(f"  Likelihood: {likelihood}")
    print("==================================================")
    
    # Run the single selected configuration
    run_simulation(problem=problem_name, likelihood=likelihood, mcmc=mcmc, sampler=sampler)