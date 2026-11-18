import os
import numpy as np
import ray
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from omegaconf import OmegaConf

from particles.core import SMC
from particles.smc_samplers import AdaptiveMCMCSequence
from hsmc.samplers import TempNsampNoRew

from problem_setup.compact_problem.likelihood_model import IPModelInexactAdaptHD, CompactModel
from hsmc.mcmc import NoisyMetropolisCal, RWRegCal, TwoStageMCMC

from experiments.gaussian_prior_discrete import GaussianPriorDiscreteExperiment

class FixedCoarseBudgetSweepExperiment(GaussianPriorDiscreteExperiment):
    
    plot_labels = ["Two-Stage Adapt", "Std Adapt", "Std Fixed"]
    plot_cost = False

    def setup(self, cfg):
        from omegaconf import open_dict
        
        problem_cfg = cfg.problem
        sweep_cfg = cfg.experiment.sweep
        
        self.n_runs = sweep_cfg.n_runs
        self.budget_seq = list(sweep_cfg.budget_seq)
        self.n_particles_seq = list(sweep_cfg.n_particles_seq)
        self.gamma_nsamp_seq = list(sweep_cfg.gamma_nsamp_seq)
        self.nsamp_min = cfg.experiment.nsamp_min
        
        self.Nmax = cfg.experiment.Nmax
        self.len_c = cfg.experiment.len_c
        
        if "len_chain_seq" not in cfg.experiment.sweep:
            with open_dict(cfg.experiment.sweep):
                cfg.experiment.sweep.len_chain_seq = [0]
        self.len_chain_seq = list(cfg.experiment.sweep.len_chain_seq)
                
        self.param_dim = problem_cfg.param_dim
        
        with open_dict(cfg.experiment.sweep):
            cfg.experiment.sweep.Nmax_seq = list(self.budget_seq)
            
        super().setup(cfg)
        
        self.budget_seq = self.Nmax_seq
        
    def generate_tasks(self, cfg):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        tasks = []
        for lc_idx, len_chain in enumerate(self.len_chain_seq):
            for gns_idx, gamma_nsamp in enumerate(self.gamma_nsamp_seq):
                for budget_idx, budget in enumerate(self.budget_seq):
                    for np_idx, np_val in enumerate(self.n_particles_seq):
                        for run_idx in range(self.n_runs):
                            tasks.append({
                                "len_chain_idx": lc_idx,
                                "gamma_nsamp": gamma_nsamp,
                                "gamma_nsamp_idx": gns_idx,
                                "n_temp_steps": self.n_temp_steps,
                                "exponents": self.fixed_exponents,
                                "budget": budget,
                                "budget_idx": budget_idx,
                                "n_particles": np_val,
                                "n_particles_idx": np_idx,
                                "run_idx": run_idx,
                                "prior_ref": self.prior_ref,
                                "observed_data_ref": self.observed_data_ref,
                                "post_samp_ref_exact_ref": self.post_samp_ref_exact_ref,
                                "cfg": cfg_dict,
                                "runner_class": self.__class__
                            })
        return tasks

    @staticmethod
    def run_task(task_params):
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
        
        budget = task_params["budget"]
        n_particles = task_params["n_particles"]
        
        n_temp_steps = task_params["n_temp_steps"]
        gamma_nsamp = task_params["gamma_nsamp"]
        exponents = task_params["exponents"]
        
        nmax = cfg.experiment.Nmax
        len_c = cfg.experiment.len_c
        nsamp_min = cfg.experiment.nsamp_min
        
        runner_class = task_params.get("runner_class", GaussianPriorDiscreteExperiment)
        sched_kwargs = cfg.experiment.get("scheduler_params", {})
        Nsamps_adapt = runner_class.get_adaptive_schedule(nmax, gamma_nsamp, n_temp_steps, nsamp_min, **sched_kwargs)
        Nsamps_fixed = nmax * np.ones(len(exponents) + 1)
        
        fine_over_coarse_nsamp = cfg.problem.fine_over_coarse_nsamp
        cost_coarse = 2.0 / fine_over_coarse_nsamp
        cost_fine = 1.0
        
        max_coarse_steps = int(np.floor(budget / cost_coarse))
        coarse_nsteps = min(len_c, max_coarse_steps)
        remaining_budget = budget - (coarse_nsteps * cost_coarse)
        fine_nsteps = int(np.floor(remaining_budget / cost_fine))
        
        len_coarse_chain = coarse_nsteps + 1
        len_fine_chain = fine_nsteps + 1
        
        len_chain_std = int(budget) + 1
        
        model_comp = IPModelInexactAdaptHD(
            prior=prior,
            data_noise_cov=cfg.problem.data_noise_var,
            data=observed_data
        )
        
        class CustomTwoStageMCMC(TwoStageMCMC):
            def calibrate(self, W, x):
                super().calibrate(W, x)
                original_target = self.coarse_target
                def wrapped_target(x_aux):
                    original_target(x_aux)
                    if hasattr(x, "e_arr"):
                        if hasattr(self.coarse_model, "last_e_eval") and self.coarse_model.last_e_eval is not None:
                            x_aux.e_arr = self.coarse_model.last_e_eval
                        else:
                            x_aux.e_arr = np.zeros_like(x.e_arr)
                self.coarse_target = wrapped_target

        move_two_stage = CustomTwoStageMCMC(
            mcmc1=NoisyMetropolisCal(),
            mcmc2=RWRegCal(eye_prop=True),
            len_chain1=len_coarse_chain,
            len_chain2=len_fine_chain,
            coarse_model=model_comp,
            fine_over_coarse=fine_over_coarse_nsamp
        )
        
        move_std = AdaptiveMCMCSequence(mcmc=RWRegCal(eye_prop=True), adaptive=False, len_chain=len_chain_std)
        
        def run_smc(model, nsamps_arr, move_kernel):
            fk = TempNsampNoRew(model=model, wastefree=False, move=move_kernel, exponents=exponents, Nsamps=nsamps_arr)
            smc = SMC(fk=fk, N=n_particles, ESSrmin=1.0, verbose=False)
            smc.run()
            smc.setup_auxiliary_weights()
            smc.resample_move()
            
            ess = np.array(smc.summaries.ESSs)
            acc_rates_list = smc.X.shared.get("acc_rates", [])
            acc_rates = np.array([np.mean(ar) for ar in acc_rates_list]) if acc_rates_list else np.array([])
            exps = np.array(smc.X.shared.get("exponents", []))
            nsamps = np.array(smc.X.shared.get("solver_nsamp", []))
            
            param_samples = np.array(smc.X.theta['param'])
            w1_vals = [wasserstein_distance(param_samples[:, i], post_samp_ref_exact[:, i]) for i in range(cfg.problem.param_dim)]
            w1_mean = np.mean(w1_vals)
            
            return w1_mean, ess, acc_rates, exps, nsamps
            
        results = {}
        results["compact_adapt"] = run_smc(model_comp, Nsamps_adapt, move_two_stage)
        results["compact_std"] = run_smc(model_comp, Nsamps_adapt, move_std)
        results["density_adapt"] = run_smc(model_comp, Nsamps_fixed, move_std)
        
        return task_params["budget_idx"], task_params["n_particles"], task_params["run_idx"], results

    def process_result(self, task_params, result):
        budget_idx, np_val, run_idx, res_dict = result
        lc_idx = task_params["len_chain_idx"]
        gns_idx = task_params["gamma_nsamp_idx"]
        np_idx = task_params["n_particles_idx"]
        
        for prob, method, key in [
            ("compact", "adapt", "compact_adapt"),
            ("compact", "std", "compact_std"),
            ("density", "adapt", "density_adapt")
        ]:
            w1_mean, ess, acc_rates, exps, nsamps = res_dict[key]
            self.results_w1[prob][method][lc_idx, gns_idx, np_idx, budget_idx, run_idx] = w1_mean
            
            n_ess = len(ess)
            self.results_stats[prob][method]["ESS"][lc_idx, gns_idx, np_idx, budget_idx, run_idx, :n_ess] = ess
            
            n_acc = len(acc_rates)
            if n_acc > 0:
                self.results_stats[prob][method]["acc_rate"][lc_idx, gns_idx, np_idx, budget_idx, run_idx, :n_acc] = acc_rates
                
            if run_idx == 0:
                n_exps = len(exps)
                self.results_stats[prob][method]["exponents"][lc_idx, gns_idx, np_idx, budget_idx, :n_exps] = exps
                n_nsamps = len(nsamps)
                self.results_stats[prob][method]["nsamps"][lc_idx, gns_idx, np_idx, budget_idx, :n_nsamps] = nsamps

    def save_results(self, output_dir):
        z_val = 1.96
        
        def get_asym_yerr(mean_vals, ci_vals):
            lower_bound = np.clip(mean_vals - ci_vals, 1e-12, None)
            return [mean_vals - lower_bound, ci_vals]
            
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = {n: default_colors[i % len(default_colors)] for i, n in enumerate(self.n_particles_seq)}
        
        save_dict = {
            "budget_seq": np.array(self.budget_seq),
            "Nmax": self.Nmax,
            "len_c": self.len_c,
            "n_particles_seq": np.array(self.n_particles_seq),
            "len_chain_seq": np.array(self.len_chain_seq),
            "gamma_nsamp_seq": np.array(self.gamma_nsamp_seq),
            "n_temp_steps": self.n_temp_steps,
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
                    n_temp_steps = self.n_temp_steps
                    suffix = f"gns_{gamma_nsamp}_nts_{n_temp_steps}"
                    
                    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
                    
                    for np_idx, np_val in enumerate(self.n_particles_seq):
                        color = colors[np_val]
                        
                        w1_ca = self.results_w1["compact"]["adapt"][lc_idx, gns_idx, np_idx]
                        mean_ca = np.mean(w1_ca, axis=1)
                        ci_ca = z_val * np.std(w1_ca, axis=1, ddof=1) / np.sqrt(self.n_runs)
                        ax1.errorbar(self.budget_seq, mean_ca, yerr=get_asym_yerr(mean_ca, ci_ca),
                                     marker='o', linestyle='-', color=color, capsize=5, linewidth=2.5, elinewidth=2.5, markersize=8)
                        
                        w1_cs = self.results_w1["compact"]["std"][lc_idx, gns_idx, np_idx]
                        mean_cs = np.mean(w1_cs, axis=1)
                        ci_cs = z_val * np.std(w1_cs, axis=1, ddof=1) / np.sqrt(self.n_runs)
                        ax1.errorbar(self.budget_seq, mean_cs, yerr=get_asym_yerr(mean_cs, ci_cs),
                                     marker='s', linestyle='--', color=color, capsize=5, linewidth=2.5, elinewidth=2.5, markersize=8)
                                     
                        w1_da = self.results_w1["density"]["adapt"][lc_idx, gns_idx, np_idx]
                        mean_da = np.mean(w1_da, axis=1)
                        ci_da = z_val * np.std(w1_da, axis=1, ddof=1) / np.sqrt(self.n_runs)
                        ax1.errorbar(self.budget_seq, mean_da, yerr=get_asym_yerr(mean_da, ci_da),
                                     marker='^', linestyle=':', color=color, capsize=5, linewidth=2.5, elinewidth=2.5, markersize=8)
                    
                    ax1.set_xscale('log'); ax1.set_yscale('log')
                    ax1.set_xticks(self.budget_seq); ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                    ax1.tick_params(axis='both', which='major', labelsize=16)
                    ax1.set_xlabel('Budget', fontsize=20)
                    ax1.set_ylabel('Mean Wasserstein-1 Distance', fontsize=20)
                    ax1.set_title(f'W1 Distance vs Budget', fontsize=20)
                    ax1.grid(True, which="both", linestyle="--", alpha=0.6)
                    
                    from matplotlib.lines import Line2D
                    alg_elements = [
                        Line2D([0], [0], color='black', marker='o', linestyle='-', linewidth=2.5, markersize=8, label=self.plot_labels[0]),
                        Line2D([0], [0], color='black', marker='s', linestyle='--', linewidth=2.5, markersize=8, label=self.plot_labels[1]),
                        Line2D([0], [0], color='black', marker='^', linestyle=':', linewidth=2.5, markersize=8, label=self.plot_labels[2])
                    ]
                    part_elements = [Line2D([0], [0], color=colors[n], linewidth=2.5, label=f'$N = {n}$') for n in self.n_particles_seq]
                    
                    leg1 = ax1.legend(handles=alg_elements, loc='upper right', fontsize=14, title="Algorithms")
                    ax1.add_artist(leg1)
                    ax1.legend(handles=part_elements, loc='lower left', fontsize=14, title="Particles")
                    
                    fig1.tight_layout()
                    fig1.savefig(os.path.join(output_dir, f'w1_distance_combined_{suffix}.pdf'))
                    plt.close(fig1)
                        
                for prob, method, key in [
                    ("compact", "adapt", "compact_adapt"),
                    ("compact", "std", "compact_std"),
                    ("density", "adapt", "density_adapt")
                ]:
                    for np_idx, np_val in enumerate(self.n_particles_seq):
                        for budget_idx, budget in enumerate(self.budget_seq):
                            stats = self.results_stats[prob][method]
                            exps = stats["exponents"][lc_idx, gns_idx, np_idx, budget_idx, :n_temp_steps+1]
                            nsamps_used = stats["nsamps"][lc_idx, gns_idx, np_idx, budget_idx, :n_temp_steps+1]
                            
                            mean_ess = np.nanmean(stats["ESS"][lc_idx, gns_idx, np_idx, budget_idx, :, :], axis=0)
                            mean_acc = np.nanmean(stats["acc_rate"][lc_idx, gns_idx, np_idx, budget_idx, :, :], axis=0)
                                    
                            mean_ess = mean_ess[~np.isnan(mean_ess)]
                            mean_acc = mean_acc[~np.isnan(mean_acc)]
                            
                            f.write("=" * 80 + "\n")
                            f.write(f"Algorithm: {key} | Budget: {budget} | Particles: {np_val} | gamma: {gamma_nsamp} | n_temp_steps: {n_temp_steps}\n")
                            f.write("=" * 95 + "\n")
                            f.write(f"{'Step':>4} | {'Target Exponent':>15} | {'Avg ESS':>10} | {'Avg Acc. Rate':>15} | {'Samples':>10}\n")
                            f.write("-" * 95 + "\n")
                            
                            n_steps = len(mean_ess)
                            for t in range(n_steps):
                                exp_val = exps[t + 1] if (t + 1) < len(exps) else 0.0
                                ess_val = mean_ess[t]
                                
                                if t < len(mean_acc):
                                    acc_val = mean_acc[t]
                                    acc_str = f"{acc_val:.3f}"
                                else:
                                    acc_str = "N/A"
                                    
                                nsamp_val = nsamps_used[t + 1] if (t + 1) < len(nsamps_used) else 0.0
                                    
                                f.write(f"{t:4d} | {exp_val:15.5f} | {ess_val:10.2f} | {acc_str:>15} | {nsamp_val:10.2f}\n")
                            f.write("\n")
            
        print(f"Stats summary saved to {stats_file}")
