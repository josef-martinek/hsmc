import os
import importlib
import itertools
import numpy as np
import ray
from particles import distributions as dists
from particles import smc_samplers as ssp
from particles.core import SMC
from hydra.utils import instantiate
from omegaconf import OmegaConf

from manager.experiment import ExperimentRunner

class GenericSweepExperiment(ExperimentRunner):
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
        gt = np.asarray(problem_cfg.ground_truth_par)
        true_sol = ModelExact.exact_sol(gt).reshape(-1, 1)

        rng_data = np.random.default_rng(seed=81)
        observed_data = true_sol + rng_data.normal(
            loc=0.,
            scale=np.sqrt(problem_cfg.data_noise_var),
            size=true_sol.shape
        )

        # 4. Put shared variables into Ray object store
        self.prior_ref = ray.put(prior)
        self.observed_data_ref = ray.put(observed_data)

        # 5. Store attributes for processing & saving
        self.observed_data = observed_data
        
        sweep_cfg = cfg.sweep
        self.n_runs = sweep_cfg.n_runs

        # Initialize lists for task data collection
        self.setups = []
        self.raw_particles_by_setup = []
        self.stats_by_setup = []

    def generate_tasks(self, cfg):
        sweep_cfg = cfg.sweep
        n_runs = sweep_cfg.n_runs
        
        # variables is a dict: {dot_path: [values]}
        sweep_vars = sweep_cfg.variables
        keys, values = zip(*sweep_vars.items())
        experiments_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Build primitive serializable cfg dict
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        tasks = []
        for run_idx in range(n_runs):
            for grid_point in experiments_grid:
                tasks.append({
                    "run_idx": run_idx,
                    "grid_point": grid_point,
                    "prior_ref": self.prior_ref,
                    "observed_data_ref": self.observed_data_ref,
                    "cfg": cfg_dict
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
        run_idx = task_params["run_idx"]
        np.random.seed(seed=run_idx)

        # 2. Retrieve shared objects from Ray store
        prior_ref = task_params["prior_ref"]
        observed_data_ref = task_params["observed_data_ref"]
        
        prior = ray.get(prior_ref)
        observed_data = ray.get(observed_data_ref)

        # 3. Read config & apply grid point overrides
        cfg = OmegaConf.create(task_params["cfg"])
        for dot_path, val in task_params["grid_point"].items():
            OmegaConf.update(cfg, dot_path, val)

        # 4. Dynamic instantiation of likelihood models via Hydra
        model = instantiate(
            cfg.likelihood,
            prior=prior,
            data_noise_cov=cfg.problem.data_noise_var,
            data=observed_data
        )

        # 5. Dynamic MCMC transition setup
        mcmc_obj = instantiate(cfg.mcmc)
        move = ssp.AdaptiveMCMCSequence(
            mcmc=mcmc_obj,
            adaptive=False,
            len_chain=cfg.problem.len_chain
        )

        # 6. Dynamic Tempering Sampler
        fk_tempering = instantiate(
            cfg.sampler,
            model=model,
            move=move,
            len_chain=cfg.problem.len_chain
        )

        # 7. Run SMC
        my_temp_alg = SMC(
            fk=fk_tempering,
            N=cfg.problem.n_particles,
            ESSrmin=1.,
            verbose=False
        )
        my_temp_alg.run()
        my_temp_alg.setup_auxiliary_weights()
        my_temp_alg.resample_move()

        post_samp = my_temp_alg.X.theta['param']
        return post_samp

    def process_result(self, task_params, result, storage_mode):
        grid_point = task_params["grid_point"]
        run_idx = task_params["run_idx"]
        setup = {"grid_point": grid_point, "run_idx": run_idx}
        self.setups.append(setup)

        # Predefine scalar statistics (mean & standard deviation per dimension)
        mean = np.mean(result, axis=0)
        std = np.std(result, axis=0)
        
        self.stats_by_setup.append({
            "grid_point": grid_point,
            "run_idx": run_idx,
            "mean": mean,
            "std": std
        })

        # Store raw particles if Regime A (raw_particles)
        if storage_mode == "raw_particles":
            self.raw_particles_by_setup.append((setup, result.astype(np.float32)))

    def save_results(self, output_dir):
        save_dict = {
            "n_runs": self.n_runs,
            "observed_data": self.observed_data,
            "stats": np.array(self.stats_by_setup, dtype=object),
            "setups": np.array(self.setups, dtype=object)
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
