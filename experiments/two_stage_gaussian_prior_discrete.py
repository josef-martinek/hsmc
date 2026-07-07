import os
import time
import numpy as np
import ray
from scipy.stats import wasserstein_distance
from omegaconf import OmegaConf

from particles import distributions as dists
from particles import smc_samplers as ssp
from particles.core import SMC

from experiments.complex_compact_gaussian_prior_smc import ComplexCompactGaussianPriorSMCExperiment
from utils.is_get_ref_sol import get_ref_sols
from problem_setup.compact_problem.likelihood_model import IPModelExactHD, IPModelInexactFixHD, IPModelInexactAdaptHD, CompactModel
from core.mcmc import NoisyMetropolisCal, RWRegCal, TwoStageMCMC

class TwoStageGaussianPriorDiscreteExperiment(ComplexCompactGaussianPriorSMCExperiment):
    def setup(self, cfg):
        problem_cfg = cfg.problem
        sweep_cfg = cfg.experiment.sweep
        
        self.n_runs = sweep_cfg.n_runs
        self.budgets = sweep_cfg.budgets
        
        if "percents" in sweep_cfg:
            self.percents_on_coarse = np.asarray(sweep_cfg.percents)
        else:
            percent_step = sweep_cfg.percent_step
            percent_values = int(100 / percent_step) + 1
            self.percents_on_coarse = np.linspace(0.0, 1.0, percent_values)

        self.param_dim = problem_cfg.param_dim
        
        CompactModel.setup(
            m=problem_cfg.m,
            d=problem_cfg.param_dim,
            truncation_radius_R=problem_cfg.truncation_radius_R,
            base_variance=problem_cfg.base_variance,
            seed=problem_cfg.ground_truth_seed
        )
        
        rng_truth = np.random.default_rng(seed=problem_cfg.ground_truth_seed)
        self.ground_truth_par = self.sample_ground_truth(problem_cfg, rng_truth)
        prior = self.get_prior(problem_cfg)
        
        true_sol = CompactModel.exact_sol(self.ground_truth_par).reshape(-1, 1)
        rng_data = np.random.default_rng(seed=81)
        observed_data = true_sol + rng_data.normal(
            loc=0., scale=np.sqrt(problem_cfg.data_noise_var), size=true_sol.shape
        )
        
        model_ex = IPModelExactHD(prior=prior, data_noise_cov=problem_cfg.data_noise_var, data=observed_data)
        
        if "tempering_sampler" not in cfg.experiment:
            raise ValueError("tempering_sampler config missing")
            
        temp_cfg = cfg.experiment.tempering_sampler
        print("Extracting e1 using single-step AdaptiveTempering...")
        np.random.seed(problem_cfg.ground_truth_seed)
        
        mcmc_temp = RWRegCal()
        move_temp = ssp.AdaptiveMCMCSequence(mcmc=mcmc_temp, adaptive=False, len_chain=temp_cfg.len_chain)
        fk_temp = ssp.AdaptiveTempering(model=model_ex, wastefree=False, ESSrmin=temp_cfg.ESSrmin, move=move_temp)
        
        alg_temp = SMC(fk=fk_temp, N=temp_cfg.N, ESSrmin=1., verbose=False)
        next(alg_temp)
        
        temp_exps = alg_temp.X.shared.get('exponents', [])
        self.e1 = temp_exps[-1]
            
        print(f"Target exponent e1: {self.e1}")
        
        print("Computing exact reference samples at e1...")
        ref_sol_cfg = cfg.experiment.ref_sol
        mcmc_ref = RWRegCal()
        move_ref = ssp.AdaptiveMCMCSequence(mcmc=mcmc_ref, adaptive=False, len_chain=ref_sol_cfg.len_chain)
        fk_ref = ssp.Tempering(model=model_ex, wastefree=False, exponents=[self.e1], move=move_ref)
        alg_ref = SMC(fk=fk_ref, N=ref_sol_cfg.N, ESSrmin=1., verbose=True)
        alg_ref.run()
        alg_ref.setup_auxiliary_weights()
        alg_ref.resample_move()
        post_samp_ref_exact = alg_ref.X.theta['param']
        print("Reference samples at e1 computed.")
        
        self.prior_ref = ray.put(prior)
        self.observed_data_ref = ray.put(observed_data)
        self.post_samp_ref_exact_ref = ray.put(post_samp_ref_exact)
        
        self.setups = []
        self.W_ex_by_setup = []

    def generate_tasks(self, cfg):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        tasks = []
        
        for budget in self.budgets:
            for percent in self.percents_on_coarse:
                for run_idx in range(self.n_runs):
                    tasks.append({
                        "budget": budget,
                        "percent": percent,
                        "run_idx": run_idx,
                        "e1": self.e1,
                        "prior_ref": self.prior_ref,
                        "observed_data_ref": self.observed_data_ref,
                        "post_samp_ref_exact_ref": self.post_samp_ref_exact_ref,
                        "cfg": cfg_dict,
                        "param_dim": self.param_dim,
                        "fine_over_coarse_nsamp": cfg.problem.fine_over_coarse_nsamp,
                        "data_noise_var": cfg.problem.data_noise_var,
                        "Nmin": cfg.problem.Nmin,
                        "n_particles": cfg.problem.n_particles
                    })
        return tasks

    @staticmethod
    def run_task(task_params):
        import numpy as np
        from omegaconf import OmegaConf
        import ray
        from particles import smc_samplers as ssp
        from particles.core import SMC
        from core.mcmc import NoisyMetropolisCal, RWRegCal, TwoStageMCMC
        from problem_setup.compact_problem.likelihood_model import IPModelInexactAdaptHD, IPModelInexactFixHD, CompactModel
        
        run_id = task_params["run_idx"]
        np.random.seed(seed=run_id)
        
        cfg = OmegaConf.create(task_params["cfg"])
        
        CompactModel.setup(
            m=cfg.problem.m,
            d=cfg.problem.param_dim,
            truncation_radius_R=cfg.problem.truncation_radius_R,
            base_variance=cfg.problem.base_variance,
            seed=cfg.problem.ground_truth_seed
        )
        
        prior = ray.get(task_params["prior_ref"])
        observed_data = ray.get(task_params["observed_data_ref"])
        
        budget = task_params["budget"]
        percent = task_params["percent"]
        fine_over_coarse_nsamp = task_params["fine_over_coarse_nsamp"]
        
        coarse_nsteps = int(np.floor(percent * budget * fine_over_coarse_nsamp / 2))
        fine_nsteps = int(np.floor(budget * (1 - percent)))
        len_coarse_chain = coarse_nsteps + 1
        len_fine_chain = fine_nsteps + 1
        
        model_fine = IPModelInexactFixHD(
            prior=prior,
            data_noise_cov=task_params["data_noise_var"],
            data=observed_data,
            nsamp=task_params["Nmin"]
        )
        
        model_coarse = IPModelInexactAdaptHD(
            prior=prior,
            data_noise_cov=task_params["data_noise_var"],
            data=observed_data
        )
        
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
        
        fk_tempering = ssp.Tempering(
            model=model_fine,
            wastefree=False,
            exponents=[task_params["e1"]],
            move=two_stage_chain
        )
        
        my_temp_alg = SMC(
            fk=fk_tempering,
            N=task_params["n_particles"],
            ESSrmin=1.,
            verbose=False
        )
        
        my_temp_alg.run()
        my_temp_alg.X.shared["solver_nsamp"] = [task_params["Nmin"]]
        my_temp_alg.setup_auxiliary_weights()
        my_temp_alg.resample_move()
        
        post_samp = my_temp_alg.X.theta['param']
        return post_samp

    def process_result(self, task_params, result):
        from scipy.stats import wasserstein_distance
        
        budget = task_params["budget"]
        percent = task_params["percent"]
        setup = [budget, percent]
        self.setups.append(setup)
        
        post_samp_ref_exact = ray.get(task_params["post_samp_ref_exact_ref"])
        
        par_dim = result.shape[1]
        W_ex = np.zeros(par_dim)
        
        for i in range(par_dim):
            W_ex[i] = wasserstein_distance(result[:, i], post_samp_ref_exact[:, i])
            
        self.W_ex_by_setup.append(np.mean(W_ex))

    def save_results(self, output_dir):
        save_dict = {
            "n_runs": self.n_runs,
            "W_ex_by_setup": np.array(self.W_ex_by_setup),
            "setups": np.array(self.setups)
        }
        out_file = os.path.join(output_dir, "smc_results.npz")
        np.savez_compressed(out_file, **save_dict)
        print(f"Compressed results successfully saved to {out_file}")
