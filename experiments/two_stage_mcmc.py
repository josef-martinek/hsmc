import os
import time
import numpy as np
import ray
from scipy.stats import wasserstein_distance
from particles import distributions as dists
from particles import smc_samplers as ssp
from particles.core import SMC
from omegaconf import OmegaConf

from manager.experiment import ExperimentRunner
from utils.is_get_ref_sol import get_ref_sols
from problem_setup.wos.likelihood_model import WOSHD, IPModelInexactFixHD, IPModelInexactAdaptHD
from core.mcmc import NoisyMetropolisCal, RWRegCal, TwoStageMCMC

class Test2StageMCMCExperiment(ExperimentRunner):
    def setup(self, cfg):
        problem_cfg = cfg.problem
        sweep_cfg = cfg.experiment.sweep
        
        # Determine the param_dims sweep list, falling back to [problem_cfg.param_dim] if not provided
        if "param_dims" in sweep_cfg:
            param_dims = list(sweep_cfg.param_dims)
        else:
            param_dims = [problem_cfg.param_dim]

        self.param_dims = param_dims
        self.WOS_ref_sol = cfg.WOS_ref_sol
        self.n_runs = sweep_cfg.n_runs

        # Dictionaries to store ray.put refs and reference solutions per param_dim
        self.prior_refs = {}
        self.observed_data_refs = {}
        self.post_samp_refs_exact = {}
        self.post_samp_refs_wos = {}
        self.observed_datas = {}

        for param_dim in param_dims:
            print(f"Setting up dimension: {param_dim}")
            
            # Prior is MvNormal as defined in the WoS experiment
            prior_marg_var = problem_cfg.prior_marg_var
            prior = dists.StructDist({
                'param': dists.MvNormal(
                    loc=np.zeros(param_dim),
                    cov=prior_marg_var * np.eye(param_dim)
                )
            })

            # Replicate exact legacy generation: seed=5, scale=4.0
            rng_gt = np.random.default_rng(seed=problem_cfg.ground_truth_seed)
            gt = rng_gt.normal(scale=4.0, size=param_dim)

            true_sol = WOSHD.exact_sol(gt).reshape(-1, 1)

            rng_data = np.random.default_rng(seed=81)
            observed_data = true_sol + rng_data.normal(
                loc=0.,
                scale=np.sqrt(problem_cfg.data_noise_var),
                size=true_sol.shape
            )

            # Compute reference solutions once per dimension
            print(f"Computing reference solutions for param_dim={param_dim}...")
            post_samp_ref_exact, post_samp_ref_wos = get_ref_sols(
                prior,
                problem_cfg.data_noise_var,
                observed_data,
                problem_cfg.Nmin,
                self.WOS_ref_sol
            )
            print(f"Reference solutions computed for param_dim={param_dim}.")

            # Put shared variables into Ray object store
            self.prior_refs[param_dim] = ray.put(prior)
            self.observed_data_refs[param_dim] = ray.put(observed_data)
            
            # Store reference solutions and observed data mapping
            self.post_samp_refs_exact[param_dim] = post_samp_ref_exact
            self.post_samp_refs_wos[param_dim] = post_samp_ref_wos
            self.observed_datas[param_dim] = observed_data

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
        for param_dim in self.param_dims:
            for budget in budgets:
                for percent in percents_on_coarse:
                    for run_idx in range(n_runs):
                        tasks.append({
                            "param_dim": param_dim,
                            "budget": budget,
                            "percent": percent,
                            "run_idx": run_idx,
                            "prior_ref": self.prior_refs[param_dim],
                            "observed_data_ref": self.observed_data_refs[param_dim],
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
        
        # Explicitly override the dimension in the configuration so workers use the correct model dimension
        cfg.problem.param_dim = task_params["param_dim"]

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

        # 5. Instantiate Models directly
        model_fine = IPModelInexactFixHD(
            prior=prior,
            data_noise_cov=data_noise_var,
            data=observed_data,
            nsamp=Nmin
        )
        model_coarse = IPModelInexactAdaptHD(
            prior=prior,
            data_noise_cov=data_noise_var,
            data=observed_data
        )

        # 6. Instantiate MCMC components directly
        mcmc_coarse = NoisyMetropolisCal()
        mcmc_fine = RWRegCal(eye_prop=True)

        two_stage_chain = TwoStageMCMC(
            mcmc1=mcmc_coarse,
            mcmc2=mcmc_fine,
            len_chain1=len_coarse_chain,
            len_chain2=len_fine_chain,
            coarse_model=model_coarse,
            fine_over_coarse=fine_over_coarse_nsamp
        )

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
        dim = task_params["param_dim"]
        setup = [budget, percent, dim]
        self.setups.append(setup)

        # Compute Wasserstein distances against references (Mean only)
        par_dim = result.shape[1]
        W_ex = np.zeros(par_dim)
        W_wos = np.zeros(par_dim)

        post_ref_exact = self.post_samp_refs_exact[dim]
        post_ref_wos = self.post_samp_refs_wos[dim]

        for i in range(par_dim):
            W_ex[i] = wasserstein_distance(result[:, i], post_ref_exact[:, i])
            W_wos[i] = wasserstein_distance(result[:, i], post_ref_wos[:, i])

        self.W_ex_by_setup.append(np.mean(W_ex))
        self.W_wos_by_setup.append(np.mean(W_wos))

        # Store raw particles if Regime A (raw_particles)
        if storage_mode == "raw_particles":
            self.raw_particles_by_setup.append((task_params, result.astype(np.float32)))

    def save_results(self, output_dir):
        save_dict = {
            "n_runs": self.n_runs,
            "param_dims": np.array(self.param_dims),
            "observed_datas": self.observed_datas,
            "W_ex_by_setup": np.array(self.W_ex_by_setup),
            "W_wos_by_setup": np.array(self.W_wos_by_setup),
            "setups": np.array(self.setups),
            "WOS_ref_sol": self.WOS_ref_sol
        }

        # If raw particles were stored, save them compressed
        if self.raw_particles_by_setup:
            particles_list = [p[1] for p in self.raw_particles_by_setup]
            params_list = [p[0] for p in self.raw_particles_by_setup]
            
            raw_particles = np.empty(len(particles_list), dtype=object)
            for idx, p in enumerate(particles_list):
                raw_particles[idx] = p
                
            raw_particles_params = np.empty(len(params_list), dtype=object)
            for idx, p in enumerate(params_list):
                raw_particles_params[idx] = p
                
            save_dict["raw_particles"] = raw_particles
            save_dict["raw_particles_params"] = raw_particles_params

        out_file = os.path.join(output_dir, "smc_results.npz")
        np.savez_compressed(out_file, **save_dict)
        print(f"Compressed results successfully saved to {out_file}")
