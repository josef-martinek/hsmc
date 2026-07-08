import numpy as np
import ray
from scipy.stats import wasserstein_distance
from omegaconf import OmegaConf

from particles.core import SMC
from core.mcmc import NoisyMetropolisCal, RWRegCal, TwoStageMCMC

from problem_setup.compact_problem.likelihood_model import IPModelInexactAdaptHD, CompactModel
from problem_setup.compact_problem_density.likelihood_model import IPModelInexactAdaptHDDensity
from problem_setup.compact_problem_density.kernel_ratio import KernelRatioEvaluator
from problem_setup.compact_problem_density.samplers import DensityTempNsampNoRew
from core.samplers import TempNsampNoRew

from experiments.gaussian_prior_discrete import GaussianPriorDiscreteExperiment

class FullTwoStageGaussianPriorDiscreteExperiment(GaussianPriorDiscreteExperiment):
    
    def setup(self, cfg):
        # We extract budget and percent_coarse to compute chain lengths
        budget = cfg.experiment.budget
        percent_coarse = cfg.experiment.percent_coarse
        fine_over_coarse_nsamp = cfg.problem.fine_over_coarse_nsamp
        
        coarse_nsteps = int(np.floor(percent_coarse * budget * fine_over_coarse_nsamp / 2))
        fine_nsteps = int(np.floor(budget * (1 - percent_coarse)))
        self.len_coarse_chain = coarse_nsteps + 1
        self.len_fine_chain = fine_nsteps + 1
        
        print("==================================================")
        print("Full Two-Stage SMC Configuration:")
        print(f"Budget: {budget}, Percent Coarse: {percent_coarse}")
        print(f"Computed chain lengths -> Coarse: {self.len_coarse_chain}, Fine: {self.len_fine_chain}")
        print("==================================================")
        
        # Inject dummy len_chain_seq so the parent class arrays size correctly
        if "len_chain_seq" not in cfg.experiment.sweep:
            from omegaconf import open_dict
            with open_dict(cfg.experiment.sweep):
                cfg.experiment.sweep.len_chain_seq = [0]
                
        super().setup(cfg)
        
    @staticmethod
    def run_task(task_params):
        # 1. Setup random seed and imports
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
        
        # 2. Extract standard adaptive sweeping parameters
        nmax = task_params["nmax"]
        nsamp_min = task_params["nsamp_min"]
        n_particles = task_params["n_particles"]
        
        n_temp_steps = task_params["n_temp_steps"]
        gamma_nsamp = task_params["gamma_nsamp"]
        exponents = task_params["exponents"]
        
        runner_class = task_params.get("runner_class", GaussianPriorDiscreteExperiment)
        sched_kwargs = cfg.experiment.get("scheduler_params", {})
        Nsamps_adapt = runner_class.get_adaptive_schedule(nmax, gamma_nsamp, n_temp_steps, nsamp_min, **sched_kwargs)
        Nsamps_fixed = nmax * np.ones(len(exponents) + 1)
        
        # 3. Two-Stage MCMC Instantiation
        budget = cfg.experiment.budget
        percent_coarse = cfg.experiment.percent_coarse
        fine_over_coarse_nsamp = cfg.problem.fine_over_coarse_nsamp
        
        coarse_nsteps = int(np.floor(percent_coarse * budget * fine_over_coarse_nsamp / 2))
        fine_nsteps = int(np.floor(budget * (1 - percent_coarse)))
        len_coarse_chain = coarse_nsteps + 1
        len_fine_chain = fine_nsteps + 1
        
        mcmc_coarse = NoisyMetropolisCal()
        mcmc_fine = RWRegCal(eye_prop=True)
        
        model_coarse = IPModelInexactAdaptHD(
            prior=prior,
            data_noise_cov=cfg.problem.data_noise_var,
            data=observed_data
        )
        
        move = TwoStageMCMC(
            mcmc1=mcmc_coarse,
            mcmc2=mcmc_fine,
            len_chain1=len_coarse_chain,
            len_chain2=len_fine_chain,
            coarse_model=model_coarse,
            fine_over_coarse=fine_over_coarse_nsamp
        )
        
        # 4. SMC runner function
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
            nsamps = np.array(smc.X.shared.get("solver_nsamp", []))
            
            param_samples = np.array(smc.X.theta['param'])
            w1_vals = [wasserstein_distance(param_samples[:, i], post_samp_ref_exact[:, i]) for i in range(cfg.problem.param_dim)]
            w1_mean = np.mean(w1_vals)
            
            return w1_mean, ess, acc_rates, exps, nsamps
            
        # 5. Run the different versions of the problem
        results = {}
        
        # Compact adapt and fixed
        m_comp_adapt = IPModelInexactAdaptHD(prior=prior, data_noise_cov=cfg.problem.data_noise_var, data=observed_data)
        results["compact_adapt"] = run_smc(m_comp_adapt, Nsamps_adapt)
        results["compact_std"] = run_smc(m_comp_adapt, Nsamps_fixed)
        
        # Compact density
        m_dens_adapt = IPModelInexactAdaptHDDensity(prior=prior, data_noise_cov=cfg.problem.data_noise_var, data=observed_data)
        m_dens_adapt.evaluator = KernelRatioEvaluator(CompactModel)
        results["density_adapt"] = run_smc(m_dens_adapt, Nsamps_adapt, sampler_class=DensityTempNsampNoRew)
        
        return task_params["nmax_idx"], task_params["n_particles"], task_params["run_idx"], results
