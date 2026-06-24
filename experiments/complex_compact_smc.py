import os
import time
import numpy as np
import ray
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from particles import distributions as dists
from particles.smc_samplers import AdaptiveMCMCSequence
from particles.core import SMC
from omegaconf import OmegaConf

from manager.experiment import ExperimentRunner
from utils.is_get_ref_sol import get_ref_sols

# Problem classes
from problem_setup.compact_problem.likelihood_model import CompactModel as ModelExact
from problem_setup.compact_problem.likelihood_model import IPModelInexactAdaptHD, IPModelInexactFixHD
from problem_setup.compact_problem_density.likelihood_model import IPModelInexactAdaptHDDensity, IPModelInexactFixHDDensity

# Samplers and MCMC
from core.samplers import TempNsampNoRew
from core.mcmc import RWRegCal


class ComplexCompactSMCExperiment(ExperimentRunner):
    def setup(self, cfg):
        problem_cfg = cfg.problem
        sweep_cfg = cfg.experiment.sweep
        
        self.n_runs = sweep_cfg.n_runs
        self.Nmax_seq = list(sweep_cfg.Nmax_seq)
        self.n_particles_seq = list(sweep_cfg.n_particles_seq)
        self.len_chain_seq = list(sweep_cfg.len_chain_seq)
        self.gamma_nsamp_seq = list(sweep_cfg.gamma_nsamp_seq)
        self.n_temp_steps_seq = list(sweep_cfg.n_temp_steps_seq)
        
        self.param_dim = problem_cfg.param_dim
        
        # ground_truth_par
        self.ground_truth_par = np.array([0.5, -0.5, 0.2, 0.8, -0.1])
        
        # prior
        a = problem_cfg.prior_a
        b = problem_cfg.prior_b
        if self.param_dim == 1:
            prior = dists.StructDist({'param': dists.Uniform(a=a, b=b)})
        else:
            prior = dists.StructDist({'param': dists.IndepProd(*(dists.Uniform(a=a, b=b) for _ in range(self.param_dim)))})
        
        # Generate observed data
        true_sol = ModelExact.exact_sol(self.ground_truth_par).reshape(-1, 1)
        rng_data = np.random.default_rng(seed=81)
        observed_data = true_sol + rng_data.normal(
            loc=0., scale=np.sqrt(problem_cfg.data_noise_var), size=true_sol.shape
        )
        
        # Compute exact posterior samples (Reference Solution)
        print("Computing exact reference solutions...")
        from problem_setup.compact_problem.likelihood_model import IPModelExactHD, CompactModel
        CompactModel.setup(
            m=problem_cfg.m,
            d=problem_cfg.param_dim,
            truncation_radius_R=problem_cfg.truncation_radius_R,
            base_variance=problem_cfg.base_variance,
            seed=problem_cfg.ground_truth_seed
        )
        model_ex = IPModelExactHD(prior=prior, data_noise_cov=problem_cfg.data_noise_var, data=observed_data)
        
        ref_sol_cfg = cfg.experiment.ref_sol
        ref_len_chain = ref_sol_cfg.len_chain
        ref_ESSrmin = ref_sol_cfg.ESSrmin
        ref_N = ref_sol_cfg.N
        
        # Seed global numpy for reproducible reference solution MCMC
        np.random.seed(problem_cfg.ground_truth_seed)
        
        post_samp_ref_exact = get_ref_sols(
            model_ex, len_chain=ref_len_chain, ESSrmin=ref_ESSrmin, N=ref_N
        )
        print("Reference solutions computed.")
        
        # Put shared variables into Ray object store
        self.prior_ref = ray.put(prior)
        self.observed_data_ref = ray.put(observed_data)
        self.post_samp_ref_exact_ref = ray.put(post_samp_ref_exact)
        
        # Dictionaries to store results: w1[problem][method][np_val] = np.zeros((len(Nmax_seq), n_runs))
        shape = (len(self.len_chain_seq), len(self.gamma_nsamp_seq), len(self.n_temp_steps_seq), len(self.n_particles_seq), len(self.Nmax_seq), self.n_runs)
        self.results_w1 = {
            "compact": {"adapt": np.zeros(shape), "std": np.zeros(shape)},
            "density": {"adapt": np.zeros(shape)}
        }
        self.results_stats = {
            "compact": {"adapt": {}, "std": {}},
            "density": {"adapt": {}}
        }
        
        max_t = max(self.n_temp_steps_seq)
        for prob, method in [("compact", "adapt"), ("compact", "std"), ("density", "adapt")]:
            self.results_stats[prob][method] = {
                "ESS": np.full((*shape, max_t + 1), np.nan),
                "acc_rate": np.full((*shape, max_t), np.nan),
                "exponents": np.full((*shape[:-1], max_t + 1), np.nan)
            }
        
        self.cost_adapt = np.zeros((len(self.len_chain_seq), len(self.gamma_nsamp_seq), len(self.n_temp_steps_seq), len(self.Nmax_seq)))
        self.cost_fixed = np.zeros((len(self.len_chain_seq), len(self.gamma_nsamp_seq), len(self.n_temp_steps_seq), len(self.Nmax_seq)))
        
        # Calculate theoretical costs based on Nmax
        gamma_temp = problem_cfg.gamma_temp
        
        for lc_idx, len_chain in enumerate(self.len_chain_seq):
            for gns_idx, gamma_nsamp in enumerate(self.gamma_nsamp_seq):
                for nts_idx, n_temp_steps in enumerate(self.n_temp_steps_seq):
                    T_exponents = n_temp_steps
                    for nmax_idx, nmax in enumerate(self.Nmax_seq):
                        # Fixed cost
                        c_fixed = nmax + nmax * (len_chain - 1) * T_exponents
                        self.cost_fixed[lc_idx, gns_idx, nts_idx, nmax_idx] = c_fixed
                        
                        # Adaptive cost
                        nsamp_sched = np.zeros(n_temp_steps + 1)
                        for t in range(n_temp_steps, -1, -1):
                            nsamp_sched[t] = nmax / (gamma_nsamp ** (n_temp_steps - t))
                        
                        c_adapt = nsamp_sched[0]
                        for t in range(1, T_exponents + 1):
                            c_adapt += nsamp_sched[t] * (len_chain - 1)
                            if nsamp_sched[t] > nsamp_sched[t-1]:
                                c_adapt += nsamp_sched[t]
                        self.cost_adapt[lc_idx, gns_idx, nts_idx, nmax_idx] = c_adapt

    def generate_tasks(self, cfg):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        tasks = []
        for lc_idx, len_chain in enumerate(self.len_chain_seq):
            for gns_idx, gamma_nsamp in enumerate(self.gamma_nsamp_seq):
                for nts_idx, n_temp_steps in enumerate(self.n_temp_steps_seq):
                    for nmax_idx, nmax in enumerate(self.Nmax_seq):
                        for np_idx, np_val in enumerate(self.n_particles_seq):
                            for run_idx in range(self.n_runs):
                                tasks.append({
                                    "len_chain": len_chain,
                                    "len_chain_idx": lc_idx,
                                    "gamma_nsamp": gamma_nsamp,
                                    "gamma_nsamp_idx": gns_idx,
                                    "n_temp_steps": n_temp_steps,
                                    "n_temp_steps_idx": nts_idx,
                                    "nmax": nmax,
                                    "nmax_idx": nmax_idx,
                                    "n_particles": np_val,
                                    "n_particles_idx": np_idx,
                                    "run_idx": run_idx,
                                    "prior_ref": self.prior_ref,
                                    "observed_data_ref": self.observed_data_ref,
                                    "post_samp_ref_exact_ref": self.post_samp_ref_exact_ref,
                                    "cfg": cfg_dict
                                })
        return tasks

    @staticmethod
    def run_task(task_params):
        import numpy as np
        from scipy.stats import wasserstein_distance
        from omegaconf import OmegaConf
        from particles.smc_samplers import AdaptiveMCMCSequence
        from particles.core import SMC
        from core.mcmc import RWRegCal
        from particles import distributions as dists
        
        from problem_setup.compact_problem.likelihood_model import IPModelInexactAdaptHD, IPModelInexactFixHD, CompactModel
        from problem_setup.compact_problem_density.likelihood_model import IPModelInexactFixHDDensity, IPModelInexactAdaptHDDensity
        from problem_setup.compact_problem_density.kernel_ratio import KernelRatioEvaluator
        from problem_setup.compact_problem_density.samplers import DensityTempNsampNoRew
        
        run_id = task_params["run_idx"]
        np.random.seed(seed=run_id)
        
        prior = ray.get(task_params["prior_ref"])
        observed_data = ray.get(task_params["observed_data_ref"])
        post_samp_ref_exact = ray.get(task_params["post_samp_ref_exact_ref"])
        cfg = OmegaConf.create(task_params["cfg"])
        
        CompactModel.setup(
            m=cfg.problem.m,
            d=cfg.problem.param_dim,
            truncation_radius_R=cfg.problem.truncation_radius_R,
            base_variance=cfg.problem.base_variance,
            seed=cfg.problem.ground_truth_seed
        )
        
        nmax = task_params["nmax"]
        n_particles = task_params["n_particles"]
        
        gamma_temp = cfg.problem.gamma_temp
        n_temp_steps = task_params["n_temp_steps"]
        gamma_nsamp = task_params["gamma_nsamp"]
        len_chain = task_params["len_chain"]
        
        exponents = np.array([(gamma_temp**t - 1) / (gamma_temp**n_temp_steps - 1) for t in range(1, n_temp_steps + 1)])
        
        Nsamps_adapt = np.zeros(n_temp_steps + 1)
        for t in range(n_temp_steps, -1, -1):
            Nsamps_adapt[t] = nmax / (gamma_nsamp ** (n_temp_steps - t))
            
        Nsamps_fixed = nmax * np.ones(len(exponents) + 1)
        
        mcmc = RWRegCal()
        move = AdaptiveMCMCSequence(mcmc=mcmc, adaptive=False, len_chain=len_chain)
        
        def run_smc(model, nsamps_arr, sampler_class=TempNsampNoRew):
            fk = sampler_class(model=model, wastefree=False, move=move, exponents=exponents, Nsamps=nsamps_arr)
            smc = SMC(fk=fk, N=n_particles, ESSrmin=1.0, verbose=False)
            smc.run()
            smc.setup_auxiliary_weights()
            smc.resample_move()
            
            ess = np.array(smc.summaries.ESSs)
            acc_rates_list = smc.X.shared.get("acc_rates", [])
            acc_rates = np.array([np.mean(ar) for ar in acc_rates_list]) if acc_rates_list else np.array([])
            exps = np.array(smc.X.shared.get("exponents", []))
            
            param_samples = np.array(smc.X.theta['param'])
            w1_vals = [wasserstein_distance(param_samples[:, i], post_samp_ref_exact[:, i]) for i in range(cfg.problem.param_dim)]
            w1_mean = np.mean(w1_vals)
            
            return w1_mean, ess, acc_rates, exps
            
        results = {}
        
        # 1. Compact
        m_comp_adapt = IPModelInexactAdaptHD(prior=prior, data_noise_cov=cfg.problem.data_noise_var, data=observed_data)
        results["compact_adapt"] = run_smc(m_comp_adapt, Nsamps_adapt)
        results["compact_std"] = run_smc(m_comp_adapt, Nsamps_fixed)
        
        # 2. Compact Density
        m_data = len(observed_data)
        a = cfg.problem.prior_a
        b = cfg.problem.prior_b
        param_dim = cfg.problem.param_dim
        if param_dim == 1:
            prior_param = dists.Uniform(a=a, b=b)
        else:
            prior_param = dists.IndepProd(*(dists.Uniform(a=a, b=b) for _ in range(param_dim)))
            
        prior_dens = dists.StructDist({'param': prior_param})
        
        m_dens_adapt = IPModelInexactAdaptHDDensity(prior=prior_dens, data_noise_cov=cfg.problem.data_noise_var, data=observed_data)
        m_dens_adapt.evaluator = KernelRatioEvaluator(CompactModel)
        results["density_adapt"] = run_smc(m_dens_adapt, Nsamps_adapt, sampler_class=DensityTempNsampNoRew)
        
        return task_params["nmax_idx"], task_params["n_particles"], task_params["run_idx"], results

    def process_result(self, task_params, result):
        nmax_idx, np_val, run_idx, res_dict = result
        lc_idx = task_params["len_chain_idx"]
        gns_idx = task_params["gamma_nsamp_idx"]
        nts_idx = task_params["n_temp_steps_idx"]
        np_idx = task_params["n_particles_idx"]
        
        for prob, method, key in [
            ("compact", "adapt", "compact_adapt"),
            ("compact", "std", "compact_std"),
            ("density", "adapt", "density_adapt")
        ]:
            w1_mean, ess, acc_rates, exps = res_dict[key]
            self.results_w1[prob][method][lc_idx, gns_idx, nts_idx, np_idx, nmax_idx, run_idx] = w1_mean
            
            n_ess = len(ess)
            self.results_stats[prob][method]["ESS"][lc_idx, gns_idx, nts_idx, np_idx, nmax_idx, run_idx, :n_ess] = ess
            
            n_acc = len(acc_rates)
            if n_acc > 0:
                self.results_stats[prob][method]["acc_rate"][lc_idx, gns_idx, nts_idx, np_idx, nmax_idx, run_idx, :n_acc] = acc_rates
                
            # Only need to store exponents once per configuration since it's identical
            if run_idx == 0:
                n_exps = len(exps)
                self.results_stats[prob][method]["exponents"][lc_idx, gns_idx, nts_idx, np_idx, nmax_idx, :n_exps] = exps

    def save_results(self, output_dir):
        z_val = 1.96
        
        def get_asym_yerr(mean_vals, ci_vals):
            lower_bound = np.clip(mean_vals - ci_vals, 1e-12, None)
            return [mean_vals - lower_bound, ci_vals]
            
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = {n: default_colors[i % len(default_colors)] for i, n in enumerate(self.n_particles_seq)}
        
        # Save npz once with the full 6D arrays
        save_dict = {
            "Nmax_seq": np.array(self.Nmax_seq),
            "n_particles_seq": np.array(self.n_particles_seq),
            "len_chain_seq": np.array(self.len_chain_seq),
            "gamma_nsamp_seq": np.array(self.gamma_nsamp_seq),
            "n_temp_steps_seq": np.array(self.n_temp_steps_seq),
            "cost_adapt": np.array(self.cost_adapt),
            "cost_fixed": np.array(self.cost_fixed)
        }
        for prob, method in [("compact", "adapt"), ("compact", "std"), ("density", "adapt")]:
            save_dict[f"w1_{prob}_{method}"] = self.results_w1[prob][method]
        out_file = os.path.join(output_dir, "smc_results.npz")
        np.savez_compressed(out_file, **save_dict)
        print(f"Results successfully saved to {out_file}")
        
        stats_file = os.path.join(output_dir, "stats_summary.txt")
        with open(stats_file, 'w') as f:
            for lc_idx, len_chain in enumerate(self.len_chain_seq):
                for gns_idx, gamma_nsamp in enumerate(self.gamma_nsamp_seq):
                    for nts_idx, n_temp_steps in enumerate(self.n_temp_steps_seq):
                        suffix = f"lc_{len_chain}_gns_{gamma_nsamp}_nts_{n_temp_steps}"
                        
                        # 1. Error Plot
                        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
                        
                        for np_idx, np_val in enumerate(self.n_particles_seq):
                            color = colors[np_val]
                            
                            # compact_adapt
                            w1_ca = self.results_w1["compact"]["adapt"][lc_idx, gns_idx, nts_idx, np_idx]
                            mean_ca = np.mean(w1_ca, axis=1)
                            ci_ca = z_val * np.std(w1_ca, axis=1, ddof=1) / np.sqrt(self.n_runs)
                            ax1.errorbar(self.Nmax_seq, mean_ca, yerr=get_asym_yerr(mean_ca, ci_ca),
                                         marker='o', linestyle='-', color=color, capsize=5, linewidth=2.5, elinewidth=2.5, markersize=8)
                            
                            # compact_std
                            w1_cs = self.results_w1["compact"]["std"][lc_idx, gns_idx, nts_idx, np_idx]
                            mean_cs = np.mean(w1_cs, axis=1)
                            ci_cs = z_val * np.std(w1_cs, axis=1, ddof=1) / np.sqrt(self.n_runs)
                            ax1.errorbar(self.Nmax_seq, mean_cs, yerr=get_asym_yerr(mean_cs, ci_cs),
                                         marker='s', linestyle='--', color=color, capsize=5, linewidth=2.5, elinewidth=2.5, markersize=8)
                                         
                            # density_adapt
                            w1_da = self.results_w1["density"]["adapt"][lc_idx, gns_idx, nts_idx, np_idx]
                            mean_da = np.mean(w1_da, axis=1)
                            ci_da = z_val * np.std(w1_da, axis=1, ddof=1) / np.sqrt(self.n_runs)
                            ax1.errorbar(self.Nmax_seq, mean_da, yerr=get_asym_yerr(mean_da, ci_da),
                                         marker='^', linestyle=':', color=color, capsize=5, linewidth=2.5, elinewidth=2.5, markersize=8)
                        
                        ax1.set_xscale('log'); ax1.set_yscale('log')
                        ax1.set_xticks(self.Nmax_seq); ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                        ax1.tick_params(axis='both', which='major', labelsize=16)
                        ax1.set_xlabel('$S_{max}$', fontsize=20)
                        ax1.set_ylabel('Mean Wasserstein-1 Distance', fontsize=20)
                        ax1.set_title(f'W1 Distance ({suffix})', fontsize=20)
                        ax1.grid(True, which="both", linestyle="--", alpha=0.6)
                        
                        from matplotlib.lines import Line2D
                        alg_elements = [
                            Line2D([0], [0], color='black', marker='o', linestyle='-', linewidth=2.5, markersize=8, label='Compact Adapt'),
                            Line2D([0], [0], color='black', marker='s', linestyle='--', linewidth=2.5, markersize=8, label='Compact Std'),
                            Line2D([0], [0], color='black', marker='^', linestyle=':', linewidth=2.5, markersize=8, label='Density Adapt')
                        ]
                        part_elements = [Line2D([0], [0], color=colors[n], linewidth=2.5, label=f'$N = {n}$') for n in self.n_particles_seq]
                        
                        leg1 = ax1.legend(handles=alg_elements, loc='upper right', fontsize=14, title="Algorithms")
                        ax1.add_artist(leg1)
                        ax1.legend(handles=part_elements, loc='lower left', fontsize=14, title="Particles")
                        
                        fig1.tight_layout()
                        fig1.savefig(os.path.join(output_dir, f'w1_distance_combined_{suffix}.pdf'))
                        plt.close(fig1)
                        
                        # 2. Cost Plot
                        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
                        
                        cost_a = self.cost_adapt[lc_idx, gns_idx, nts_idx]
                        cost_f = self.cost_fixed[lc_idx, gns_idx, nts_idx]
                        
                        ax2.plot(self.Nmax_seq, cost_a, marker='o', linestyle='-', color='black', linewidth=2.5, markersize=8, label='Adaptive')
                        ax2.plot(self.Nmax_seq, cost_f, marker='s', linestyle='--', color='black', linewidth=2.5, markersize=8, label='Standard')
                        
                        anchor_nmax = self.Nmax_seq[-1]
                        anchor_cost = cost_a[-1]
                        slope_nmax = [anchor_cost * (n / anchor_nmax) for n in self.Nmax_seq]
                        slope_tnmax = [anchor_cost * (n / anchor_nmax) * len_chain for n in self.Nmax_seq]
                        
                        ax2.plot(self.Nmax_seq, slope_nmax, linestyle=':', color='black', linewidth=2, label=r'$\mathcal{O}(S_{max})$')
                        ax2.plot(self.Nmax_seq, slope_tnmax, linestyle=':', color='black', linewidth=2, label=r'$\mathcal{O}(T \cdot S_{max})$')
    
                        ax2.set_xscale('log'); ax2.set_yscale('log')
                        ax2.set_xticks(self.Nmax_seq); ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                        ax2.tick_params(axis='both', which='major', labelsize=16)
                        ax2.set_xlabel('$S_{max}$', fontsize=20)
                        ax2.set_ylabel('Cost per particle', fontsize=20)
                        ax2.set_title(f'Cost per particle ({suffix})', fontsize=20)
                        ax2.grid(True, which="both", linestyle="--", alpha=0.6)
                        ax2.legend(fontsize=16)
    
                        fig2.tight_layout()
                        fig2.savefig(os.path.join(output_dir, f'cost_combined_{suffix}.pdf'))
                        plt.close(fig2)
                        
                        # 3. Write Stats Summary block
                        for prob, method, key in [
                            ("compact", "adapt", "compact_adapt"),
                            ("compact", "std", "compact_std"),
                            ("density", "adapt", "density_adapt")
                        ]:
                            for np_idx, np_val in enumerate(self.n_particles_seq):
                                for nmax_idx, nmax in enumerate(self.Nmax_seq):
                                    stats = self.results_stats[prob][method]
                                    exps = stats["exponents"][lc_idx, gns_idx, nts_idx, np_idx, nmax_idx, :n_temp_steps+1]
                                    
                                    # Calculate means over runs, ignoring NaNs
                                    mean_ess = np.nanmean(stats["ESS"][lc_idx, gns_idx, nts_idx, np_idx, nmax_idx, :, :], axis=0)
                                    mean_acc = np.nanmean(stats["acc_rate"][lc_idx, gns_idx, nts_idx, np_idx, nmax_idx, :, :], axis=0)
                                    
                                    # Remove trailing NaNs
                                    mean_ess = mean_ess[~np.isnan(mean_ess)]
                                    mean_acc = mean_acc[~np.isnan(mean_acc)]
                                    
                                    f.write("=" * 80 + "\n")
                                    f.write(f"Algorithm: {key} | Nmax: {nmax} | Particles: {np_val} | len_chain: {len_chain} | gamma: {gamma_nsamp} | n_temp_steps: {n_temp_steps}\n")
                                    f.write("=" * 80 + "\n")
                                    f.write(f"{'Step':>4} | {'Target Exponent':>15} | {'Avg ESS':>10} | {'Avg Acc. Rate':>15}\n")
                                    f.write("-" * 80 + "\n")
                                    
                                    n_steps = len(mean_ess)
                                    for t in range(n_steps):
                                        exp_val = exps[t + 1] if exps is not None else 0.0
                                        ess_val = mean_ess[t]
                                        
                                        if t == 0:
                                            acc_str = "N/A"
                                        else:
                                            acc_val = mean_acc[t - 1]
                                            acc_str = f"{acc_val:.3f}"
                                            
                                        f.write(f"{t:4d} | {exp_val:15.5f} | {ess_val:10.2f} | {acc_str:>15}\n")
                                    f.write("\n")
            
        print(f"Stats summary saved to {stats_file}")
