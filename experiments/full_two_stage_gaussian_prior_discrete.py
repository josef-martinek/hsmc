import numpy as np
import ray
from scipy.stats import wasserstein_distance
from omegaconf import OmegaConf

from particles.core import SMC
from core.mcmc import NoisyMetropolisCal, RWRegCal, TwoStageMCMC

from problem_setup.compact_problem.likelihood_model import IPModelInexactAdaptHD, CompactModel
from core.samplers import TempNsampNoRew

from experiments.gaussian_prior_discrete import GaussianPriorDiscreteExperiment

class FullTwoStageGaussianPriorDiscreteExperiment(GaussianPriorDiscreteExperiment):
    
    # Custom labels for the 3 SMC algorithm variants in the plots
    plot_labels = ["Two-Stage Adapt", "Std Adapt", "Std Fixed"]
    plot_cost = False
    
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
        
        # 3. MCMC and Run setup
        budget = cfg.experiment.budget
        percent_coarse = cfg.experiment.percent_coarse
        fine_over_coarse_nsamp = cfg.problem.fine_over_coarse_nsamp
        
        coarse_nsteps = int(np.floor(percent_coarse * budget * fine_over_coarse_nsamp / 2))
        fine_nsteps = int(np.floor(budget * (1 - percent_coarse)))
        len_coarse_chain = coarse_nsteps + 1
        len_fine_chain = fine_nsteps + 1
        
        # Option A selected: ensure standard variants have identical overall MCMC steps
        len_chain_std = int(budget) + 1
        
        model_comp = IPModelInexactAdaptHD(
            prior=prior,
            data_noise_cov=cfg.problem.data_noise_var,
            data=observed_data
        )
        
        # Move 1: Two-Stage Adaptive
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
            coarse_model=model_comp, # coarse model is identical in this case
            fine_over_coarse=fine_over_coarse_nsamp
        )
        
        # Move 2: Standard MCMC
        from particles.smc_samplers import AdaptiveMCMCSequence
        move_std = AdaptiveMCMCSequence(mcmc=RWRegCal(eye_prop=True), adaptive=False, len_chain=len_chain_std)
        
        # 4. SMC runner function
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
            
        # 5. Run the different versions of the problem
        results = {}
        
        # To reuse the parent class's plotting seamlessly, we map our keys to the original structure:
        # compact_adapt -> Two Stage Adaptive
        # compact_std -> Standard MCMC Adaptive
        # density_adapt -> Standard MCMC Fixed
        results["compact_adapt"] = run_smc(model_comp, Nsamps_adapt, move_two_stage)
        results["compact_std"] = run_smc(model_comp, Nsamps_adapt, move_std)
        results["density_adapt"] = run_smc(model_comp, Nsamps_fixed, move_std)
        
        return task_params["nmax_idx"], task_params["n_particles"], task_params["run_idx"], results
