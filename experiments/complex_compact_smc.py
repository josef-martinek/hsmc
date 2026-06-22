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
        
        self.param_dim = problem_cfg.param_dim
        self.len_chain = problem_cfg.len_chain
        
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
        self.post_samp_ref_exact = post_samp_ref_exact
        
        # Dictionaries to store results: w1[problem][method][np_val] = np.zeros((len(Nmax_seq), n_runs))
        self.results_w1 = {
            "compact": {"adapt": {}, "std": {}},
            "density": {"adapt": {}}
        }
        for np_val in self.n_particles_seq:
            self.results_w1["compact"]["adapt"][np_val] = np.zeros((len(self.Nmax_seq), self.n_runs))
            self.results_w1["compact"]["std"][np_val] = np.zeros((len(self.Nmax_seq), self.n_runs))
            self.results_w1["density"]["adapt"][np_val] = np.zeros((len(self.Nmax_seq), self.n_runs))
        
        self.cost_adapt = []
        self.cost_fixed = []
        
        # Calculate theoretical costs based on Nmax
        gamma_temp = problem_cfg.gamma_temp
        n_temp_steps = problem_cfg.n_temp_steps
        gamma_nsamp = problem_cfg.gamma_nsamp
        
        T_exponents = n_temp_steps
        for nmax in self.Nmax_seq:
            # Fixed cost
            c_fixed = nmax + nmax * (self.len_chain - 1) * T_exponents
            self.cost_fixed.append(c_fixed)
            
            # Adaptive cost
            nsamp_sched = np.zeros(n_temp_steps + 1)
            for t in range(n_temp_steps, -1, -1):
                nsamp_sched[t] = nmax / (gamma_nsamp ** (n_temp_steps - t))
            
            c_adapt = nsamp_sched[0]
            for t in range(1, T_exponents + 1):
                c_adapt += nsamp_sched[t] * (self.len_chain - 1)
                if nsamp_sched[t] > nsamp_sched[t-1]:
                    c_adapt += nsamp_sched[t]
            self.cost_adapt.append(c_adapt)

    def generate_tasks(self, cfg):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        tasks = []
        for i, nmax in enumerate(self.Nmax_seq):
            for np_val in self.n_particles_seq:
                for run_idx in range(self.n_runs):
                    tasks.append({
                        "nmax": nmax,
                        "nmax_idx": i,
                        "n_particles": np_val,
                        "run_idx": run_idx,
                        "prior_ref": self.prior_ref,
                        "observed_data_ref": self.observed_data_ref,
                        "cfg": cfg_dict
                    })
        return tasks

    @staticmethod
    def run_task(task_params):
        import numpy as np
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
        n_temp_steps = cfg.problem.n_temp_steps
        gamma_nsamp = cfg.problem.gamma_nsamp
        len_chain = cfg.problem.len_chain
        
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
            return np.array(smc.X.theta['param'])
            
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
        
        def calc_w1(samples):
            w1_vals = [wasserstein_distance(samples[:, i], self.post_samp_ref_exact[:, i]) for i in range(self.param_dim)]
            return np.mean(w1_vals)
            
        self.results_w1["compact"]["adapt"][np_val][nmax_idx, run_idx] = calc_w1(res_dict["compact_adapt"])
        self.results_w1["compact"]["std"][np_val][nmax_idx, run_idx] = calc_w1(res_dict["compact_std"])
        self.results_w1["density"]["adapt"][np_val][nmax_idx, run_idx] = calc_w1(res_dict["density_adapt"])

    def save_results(self, output_dir):
        z_val = 1.96
        
        def get_asym_yerr(mean_vals, ci_vals):
            lower_bound = np.clip(mean_vals - ci_vals, 1e-12, None)
            return [mean_vals - lower_bound, ci_vals]
            
        colors = {50: 'orange', 100: 'blue', 200: 'green'}
        
        # 1. Error Plot
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        
        for np_val in self.n_particles_seq:
            color = colors[np_val]
            
            # compact_adapt
            w1_ca = self.results_w1["compact"]["adapt"][np_val]
            mean_ca = np.mean(w1_ca, axis=1)
            ci_ca = z_val * np.std(w1_ca, axis=1, ddof=1) / np.sqrt(self.n_runs)
            ax1.errorbar(self.Nmax_seq, mean_ca, yerr=get_asym_yerr(mean_ca, ci_ca),
                         marker='o', linestyle='-', color=color, capsize=5, linewidth=2.5, elinewidth=2.5, markersize=8)
            
            # compact_std
            w1_cs = self.results_w1["compact"]["std"][np_val]
            mean_cs = np.mean(w1_cs, axis=1)
            ci_cs = z_val * np.std(w1_cs, axis=1, ddof=1) / np.sqrt(self.n_runs)
            ax1.errorbar(self.Nmax_seq, mean_cs, yerr=get_asym_yerr(mean_cs, ci_cs),
                         marker='s', linestyle='--', color=color, capsize=5, linewidth=2.5, elinewidth=2.5, markersize=8)
                         
            # density_adapt
            w1_da = self.results_w1["density"]["adapt"][np_val]
            mean_da = np.mean(w1_da, axis=1)
            ci_da = z_val * np.std(w1_da, axis=1, ddof=1) / np.sqrt(self.n_runs)
            ax1.errorbar(self.Nmax_seq, mean_da, yerr=get_asym_yerr(mean_da, ci_da),
                         marker='^', linestyle=':', color=color, capsize=5, linewidth=2.5, elinewidth=2.5, markersize=8)
        
        ax1.set_xscale('log'); ax1.set_yscale('log')
        ax1.set_xticks(self.Nmax_seq); ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.set_xlabel('$S_{max}$', fontsize=20)
        ax1.set_ylabel('Mean Wasserstein-1 Distance', fontsize=20)
        ax1.set_title('W1 Distance - SMC Samplers', fontsize=20)
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
        fig1.savefig(os.path.join(output_dir, 'w1_distance_combined.pdf'))
        
        # 2. Cost Plot
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        
        ax2.plot(self.Nmax_seq, self.cost_adapt, marker='o', linestyle='-', color='black', linewidth=2.5, markersize=8, label='Adaptive')
        ax2.plot(self.Nmax_seq, self.cost_fixed, marker='s', linestyle='--', color='black', linewidth=2.5, markersize=8, label='Standard')
        
        anchor_nmax = self.Nmax_seq[-1]
        anchor_cost = self.cost_adapt[-1]
        slope_nmax = [anchor_cost * (n / anchor_nmax) for n in self.Nmax_seq]
        slope_tnmax = [anchor_cost * (n / anchor_nmax) * self.len_chain for n in self.Nmax_seq]
        
        ax2.plot(self.Nmax_seq, slope_nmax, linestyle=':', color='black', linewidth=2, label=r'$\mathcal{O}(S_{max})$')
        ax2.plot(self.Nmax_seq, slope_tnmax, linestyle=':', color='black', linewidth=2, label=r'$\mathcal{O}(T \cdot S_{max})$')

        ax2.set_xscale('log'); ax2.set_yscale('log')
        ax2.set_xticks(self.Nmax_seq); ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.set_xlabel('$S_{max}$', fontsize=20)
        ax2.set_ylabel('Cost per particle', fontsize=20)
        ax2.set_title('Cost per particle', fontsize=20)
        ax2.grid(True, which="both", linestyle="--", alpha=0.6)
        ax2.legend(fontsize=16)

        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'cost_combined.pdf'))
        
        # Save npz
        save_dict = {
            "Nmax_seq": np.array(self.Nmax_seq),
            "n_particles_seq": np.array(self.n_particles_seq),
            "cost_adapt": np.array(self.cost_adapt),
            "cost_fixed": np.array(self.cost_fixed)
        }
        for np_val in self.n_particles_seq:
            save_dict[f"w1_compact_adapt_{np_val}"] = self.results_w1["compact"]["adapt"][np_val]
            save_dict[f"w1_compact_std_{np_val}"] = self.results_w1["compact"]["std"][np_val]
            save_dict[f"w1_density_adapt_{np_val}"] = self.results_w1["density"]["adapt"][np_val]
                    
        out_file = os.path.join(output_dir, "smc_results.npz")
        np.savez_compressed(out_file, **save_dict)
        print(f"Results successfully saved to {out_file}")
