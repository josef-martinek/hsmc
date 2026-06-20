import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import time
from particles import distributions as dists
from particles.smc_samplers import AdaptiveMCMCSequence
from particles.core import SMC

# ==============================================================================
# PROBLEM SELECTION (Dedicated to Compact Problem)
# ==============================================================================

from problem_setup.compact_problem.problem_parameters import SamplerConfig as sc
from problem_setup.compact_problem.likelihood_model import CompactModel as ModelExact
# from problem_setup.compact_problem.likelihood_model import IPModelExactHD as LikelihoodModel
# To use inexact likelihood for Compact, comment the line above and uncomment one of these:
# from problem_setup.compact_problem.likelihood_model import IPModelInexactAdaptHD as LikelihoodModel
from problem_setup.compact_problem.likelihood_model import IPModelInexactFixHD as LikelihoodModel

# ==============================================================================
# MCMC & SAMPLER SELECTION
# ==============================================================================

# MCMC mutations (available options: RWRegCal, RWUnc, NoisyMetropolisCal)
from core.mcmc import RWRegCal as MCMCClass
# from core.mcmc import RWUnc as MCMCClass
# from core.mcmc import NoisyMetropolisCal as MCMCClass

# Samplers (available options: AdaptiveTempering, TempAdaptNsampNoRew, Tempering)
from particles.smc_samplers import AdaptiveTempering as SamplerClass
# from core.samplers import TempAdaptNsampNoRew as SamplerClass

# ==============================================================================
# RUN SETTINGS
# ==============================================================================
resample_move_end = True

if __name__ == "__main__":
    print("==================================================")
    print("Running SMC Simulation (Compact Problem)")
    print(f"  Problem:    {sc.problem}")
    print("==================================================")

    # 1. PARAMETERS SETUP
    param_dim = sc.param_dim
    ground_truth_par = np.asarray(sc.ground_truth_par)
    
    n_particles = sc.n_particles
    len_chain = sc.len_chain
    ESS_rmin = sc.ESS_rmin
    ar_tol = sc.ar_tol
    Nmin = sc.Nmin

    # 2. PRIOR DISTRIBUTION SETUP
    # Strictly enforce a uniform prior across all dimensions based on SamplerConfig
    a = sc.prior_a
    b = sc.prior_b
    if param_dim == 1:
        prior = dists.StructDist({'param': dists.Uniform(a=a, b=b)})
    else:
        prior = dists.StructDist({'param': dists.IndepProd(*(dists.Uniform(a=a, b=b) for _ in range(param_dim)))})

    # 3. SYNTHETIC DATA GENERATION
    true_sol = ModelExact.exact_sol(ground_truth_par).reshape(-1, 1)
    rng_data = np.random.default_rng(seed=81)
    observed_data = true_sol + rng_data.normal(loc=0., scale=np.sqrt(sc.data_noise_var), size=true_sol.shape)
    print(f"Observed data generated using exact solver.")

    # 4. LIKELIHOOD MODEL INSTANTIATION
    if "Fix" in LikelihoodModel.__name__:
        model = LikelihoodModel(
            prior=prior,
            data_noise_cov=sc.data_noise_var,
            data=observed_data,
            nsamp=sc.nsamp
        )
    else:
        model = LikelihoodModel(
            prior=prior,
            data_noise_cov=sc.data_noise_var,
            data=observed_data
        )

    # 5. MCMC & SAMPLER INSTANTIATION
    mcmc_obj = MCMCClass()
    move = AdaptiveMCMCSequence(mcmc=mcmc_obj, adaptive=False, len_chain=len_chain)
    
    sampler_name = SamplerClass.__name__
    if sampler_name == "AdaptiveTempering":
        fk_tempering = SamplerClass(model=model, move=move, wastefree=False, ESSrmin=ESS_rmin, len_chain=len_chain)
    elif sampler_name == "TempAdaptNsampNoRew":
        fk_tempering = SamplerClass(model=model, move=move, wastefree=False, ESSrmin=ESS_rmin, Nmin=Nmin, ar_tol=ar_tol, len_chain=len_chain)
    else:  # Standard Tempering or others
        fk_tempering = SamplerClass(model=model, move=move, len_chain=len_chain)

    # 6. RUN SIMULATION
    my_temp_alg = SMC(fk=fk_tempering, N=n_particles, ESSrmin=1., verbose=True)
    
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
    print("Results saved to data/experiment/")
    
    # Plot results
    if param_dim == 1:
        import scipy.stats as stats
        from matplotlib import pyplot as plt
        import seaborn as sb

        # Define prior distribution
        prior_dist = stats.uniform(loc=a, scale=b-a)

        # Plot results
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot histogram of sampled parameters
        if resample_move_end:
            sb.histplot(x=post_samp.flatten(), stat='density', label='SMC Samples', ax=ax)
        else:
            sb.histplot(x=post_samp.flatten(), weights=my_temp_alg.W, stat='density', label='SMC Samples', ax=ax)

        # Plot prior distribution
        x_vals = np.linspace(a - 0.5, b + 0.5, 1000)
        ax.plot(x_vals, prior_dist.pdf(x_vals), label='Prior', linestyle='dashed', color='black')

        ax.set_xlabel('param')
        ax.legend()
        
        # Save the plot
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/posterior_distribution.png", dpi=300, bbox_inches='tight')
        print("Plot saved to plots/posterior_distribution.png")
        plt.show()
    else:
        import pandas as pd
        import seaborn as sb
        from matplotlib import pyplot as plt

        # Create a DataFrame from the posterior samples
        columns = [f'param_{i+1}' for i in range(param_dim)]
        df = pd.DataFrame(post_samp, columns=columns)
        
        # Plot using seaborn pairplot
        grid = sb.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.6})
        
        # Denote the ground truth with a red point/line
        for i in range(param_dim):
            for j in range(param_dim):
                ax = grid.axes[i, j]
                if i != j:
                    ax.plot(ground_truth_par[j], ground_truth_par[i], 'ro', markersize=8, label='Ground Truth' if (i == 0 and j == 1) else "")
                else:
                    ax.axvline(ground_truth_par[i], color='red', linestyle='--', linewidth=2, label='Ground Truth' if i == 0 else "")
        
        plt.show()
