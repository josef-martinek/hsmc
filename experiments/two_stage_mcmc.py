import os
import time
import importlib
import numpy as np
import ray
from scipy.stats import wasserstein_distance
from particles import distributions as dists
from particles import smc_samplers as ssp
from particles.core import SMC
from hydra.utils import instantiate
from omegaconf import OmegaConf

from manager.experiment import ExperimentRunner
from utils.is_get_ref_sol import get_ref_sols

class Test2StageMCMCExperiment(ExperimentRunner):
    def setup(self, cfg):
        problem_cfg = cfg.problem
        param_dim = problem_cfg.param_dim
        
        # 1. Determine prior type dynamically
        prior_type = problem_cfg.prior_type
        if prior_type == "uniform":
            a = problem_cfg.prior_a
            b = problem_cfg.prior_b
            prior = dists.StructDist({
                'param': dists.Uniform(
                    a=a * np.ones(param_dim),
                    b=b * np.ones(param_dim)
                )
            })
        elif prior_type == "mvnormal":
            prior_marg_var = problem_cfg.prior_marg_var
            prior = dists.StructDist({
                'param': dists.MvNormal(
                    loc=np.zeros(param_dim),
                    cov=prior_marg_var * np.eye(param_dim)
                )
            })
        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")

        # 2. Dynamic import of Exact Model
        prob_lik = importlib.import_module(f"{problem_cfg.path}.likelihood_model")
        ModelExact = getattr(prob_lik, problem_cfg.model_exact_class)

        # 3. Generate synthetic observed data
        gt_cfg = problem_cfg.ground_truth_par
        if gt_cfg is None:
            # Replicate exact legacy generation: seed=5, scale=4.0
            rng_gt = np.random.default_rng(seed=problem_cfg.ground_truth_seed)
            gt = rng_gt.normal(scale=4.0, size=param_dim)
        else:
            gt = np.asarray(gt_cfg)

        true_sol = ModelExact.exact_sol(gt).reshape(-1, 1)

        rng_data = np.random.default_rng(seed=81)
        observed_data = true_sol + rng_data.normal(
            loc=0.,
            scale=np.sqrt(problem_cfg.data_noise_var),
            size=true_sol.shape
        )

        # 4. Compute reference solutions once
        WOS_ref_sol = cfg.WOS_ref_sol
        print("Computing reference solutions...")
        post_samp_ref_exact, post_samp_ref_wos = get_ref_sols(
            prior,
            problem_cfg.data_noise_var,
            observed_data,
            problem_cfg.Nmin,
            WOS_ref_sol
        )
        print("Reference solutions computed.")

        # 6. Put shared variables into Ray object store
        self.prior_ref = ray.put(prior)
        self.observed_data_ref = ray.put(observed_data)

        # 7. Store attributes for processing & saving
        self.param_dim = param_dim
        self.post_samp_ref_exact = post_samp_ref_exact
        self.post_samp_ref_wos = post_samp_ref_wos
        self.observed_data = observed_data
        self.WOS_ref_sol = WOS_ref_sol
        
        sweep_cfg = cfg.experiment.sweep
        self.n_runs = sweep_cfg.n_runs

        # Initialize lists for task data collection (Max, Weighted, and Initial errors removed)
        self.setups = []
        self.W_ex_by_setup = []
        self.W_wos_by_setup = []
        self.raw_particles_by_setup = []

    def generate_tasks(self, cfg):
        sweep_cfg = cfg.experiment.sweep
        n_runs = sweep_cfg.n_runs
        budgets = sweep_cfg.budgets
        
        if "percents" in sweep_cfg:
            percents_on_coarse = np.asarray(sweep_cfg.percents)
        else:
            percent_step = sweep_cfg.percent_step
            percent_values = int(100 / percent_step) + 1
            percents_on_coarse = np.linspace(0.0, 1.0, percent_values)

        # Build primitive serializable cfg dict
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        tasks = []
        for budget in budgets:
            for percent in percents_on_coarse:
                for run_idx in range(n_runs):
                    tasks.append({
                        "budget": budget,
                        "percent": percent,
                        "run_idx": run_idx,
                        "prior_ref": self.prior_ref,
                        "observed_data_ref": self.observed_data_ref,
                        "cfg": cfg_dict,
                        "Nmin": cfg.problem.Nmin,
                        "n_particles": cfg.problem.n_particles,
                        "fine_over_coarse_nsamp": cfg.problem.fine_over_coarse_nsamp,
                        "data_noise_var": cfg.problem.data_noise_var
                    })
        return tasks

    @staticmethod
    def run_task(task_params):
        import numpy as np
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from particles import smc_samplers as ssp
        from particles.core import SMC
        import ray

        # 1. Set seed
        run_id = task_params["run_idx"]
        np.random.seed(seed=run_id)

        # 2. Retrieve shared objects from Ray store
        prior_ref = task_params["prior_ref"]
        observed_data_ref = task_params["observed_data_ref"]
        
        prior = ray.get(prior_ref)
        observed_data = ray.get(observed_data_ref)

        # 3. Read config & parameters
        cfg = OmegaConf.create(task_params["cfg"])
        Nmin = task_params["Nmin"]
        n_particles = task_params["n_particles"]
        fine_over_coarse_nsamp = task_params["fine_over_coarse_nsamp"]
        data_noise_var = task_params["data_noise_var"]
        
        budget = task_params["budget"]
        percent = task_params["percent"]

        # 4. Chain length calculations
        coarse_nsteps = int(np.floor(percent * budget * fine_over_coarse_nsamp / 2))
        fine_nsteps = int(np.floor(budget * (1 - percent)))
        len_coarse_chain = coarse_nsteps + 1
        len_fine_chain = fine_nsteps + 1

        # 5. Instantiate Models via Hydra
        model_fine_cfg = {
            "_target_": f"{cfg.problem.path}.likelihood_model.IPModelInexactFix{cfg.problem.likelihood_suffix}",
            "nsamp": Nmin
        }
        model_coarse_cfg = {
            "_target_": f"{cfg.problem.path}.likelihood_model.IPModelInexactAdapt{cfg.problem.likelihood_suffix}"
        }
        
        model_fine = instantiate(
            model_fine_cfg,
            prior=prior,
            data_noise_cov=data_noise_var,
            data=observed_data
        )
        model_coarse = instantiate(
            model_coarse_cfg,
            prior=prior,
            data_noise_cov=data_noise_var,
            data=observed_data
        )

        # 6. Instantiate MCMC components
        mcmc_coarse_cfg = {"_target_": "core.mcmc.NoisyMetropolisCal"}
        mcmc_fine_cfg = {"_target_": "core.mcmc.RWRegCal", "eye_prop": True}

        mcmc_coarse = instantiate(mcmc_coarse_cfg)
        mcmc_fine = instantiate(mcmc_fine_cfg)

        two_stage_cfg = {
            "_target_": "core.mcmc.TwoStageMCMC",
            "mcmc1": mcmc_coarse,
            "mcmc2": mcmc_fine,
            "len_chain1": len_coarse_chain,
            "len_chain2": len_fine_chain,
            "coarse_model": model_coarse,
            "fine_over_coarse": fine_over_coarse_nsamp
        }
        two_stage_chain = instantiate(two_stage_cfg)

        # 7. Instantiate Tempering Sampler
        fk_tempering = ssp.Tempering(
            model=model_fine,
            wastefree=False,
            exponents=[1.],
            move=two_stage_chain
        )

        # 8. Run SMC
        my_temp_alg = SMC(
            fk=fk_tempering,
            N=n_particles,
            ESSrmin=1.,
            verbose=False
        )
        my_temp_alg.run()
        my_temp_alg.X.shared["solver_nsamp"] = [Nmin]
        my_temp_alg.setup_auxiliary_weights()
        my_temp_alg.resample_move()

        post_samp = my_temp_alg.X.theta['param']
        return post_samp

    def process_result(self, task_params, result, storage_mode):
        budget = task_params["budget"]
        percent = task_params["percent"]
        setup = [budget, percent]
        self.setups.append(setup)

        # Compute Wasserstein distances against references
        par_dim = result.shape[1]
        W_ex = np.zeros(par_dim)
        W_wos = np.zeros(par_dim)

        for i in range(par_dim):
            W_ex[i] = wasserstein_distance(result[:, i], self.post_samp_ref_exact[:, i])
            W_wos[i] = wasserstein_distance(result[:, i], self.post_samp_ref_wos[:, i])

        self.W_ex_by_setup.append(np.mean(W_ex))
        self.W_wos_by_setup.append(np.mean(W_wos))
        self.Wmax_ex_by_setup.append(np.max(W_ex))
        self.Wmax_wos_by_setup.append(np.max(W_wos))
        
        # Weighted Wasserstein distances
        weights = 1 / np.arange(1, par_dim + 1) ** self.q
        self.Ww_ex_by_setup.append(np.average(W_ex, weights=weights))
        self.Ww_wos_by_setup.append(np.average(W_wos, weights=weights))

        # Store raw particles if Regime A (raw_particles)
        if storage_mode == "raw_particles":
            self.raw_particles_by_setup.append((task_params, result.astype(np.float32)))

    def save_results(self, output_dir):
        save_dict = {
            "n_runs": self.n_runs,
            "observed_data": self.observed_data,
            "W_ex_by_setup": np.array(self.W_ex_by_setup),
            "W_wos_by_setup": np.array(self.W_wos_by_setup),
            "Wmax_ex_by_setup": np.array(self.Wmax_ex_by_setup),
            "Wmax_wos_by_setup": np.array(self.Wmax_wos_by_setup),
            "Ww_ex_by_setup": np.array(self.Ww_ex_by_setup),
            "Ww_wos_by_setup": np.array(self.Ww_wos_by_setup),
            "init_Wmean_ex": self.init_Wmean_ex,
            "init_Wmax_ex": self.init_Wmax_ex,
            "init_Ww_ex": self.init_Ww_ex,
            "setups": np.array(self.setups),
            "WOS_ref_sol": self.WOS_ref_sol
        }

        # If raw particles were stored, save them compressed
        if self.raw_particles_by_setup:
            particles_list = [p[1] for p in self.raw_particles_by_setup]
            params_list = [p[0] for p in self.raw_particles_by_setup]
            save_dict["raw_particles"] = np.array(particles_list, dtype=object)
            save_dict["raw_particles_params"] = np.array(params_list, dtype=object)

        out_file = os.path.join(output_dir, "smc_results.npz")
        np.savez_compressed(out_file, **save_dict)
        print(f"Compressed results successfully saved to {out_file}")
