import numpy as np
from particles import smc_samplers as ssp
from particles.core import SMC
from core.mcmc import RWRegCal, NoisyMetropolisCal, TwoStageMCMC

def run_single_is_trial(data_noise_var, prior, Nmin, nparticles, fine_over_coarse_nsamp, run_id, observed_data, budget, percent):
    from problem_setup.wos.likelihood_model import IPModelInexactFixHD, IPModelInexactAdaptHD

    np.random.seed(seed=run_id)

    coarse_nsteps = int(np.floor(percent * budget * fine_over_coarse_nsamp / 2))
    fine_nsteps = int(np.floor(budget * (1 - percent)))
    len_coarse_chain = coarse_nsteps + 1
    len_fine_chain = fine_nsteps + 1

    model_fine = IPModelInexactFixHD(prior=prior,
                                     data_noise_cov=data_noise_var,
                                     data=observed_data,
                                     nsamp=Nmin)
    model_coarse = IPModelInexactAdaptHD(prior=prior,
                                         data_noise_cov=data_noise_var,
                                         data=observed_data)
    mcmc_coarse = NoisyMetropolisCal()
    mcmc_fine = RWRegCal(eye_prop=True)
    two_stage_chain = TwoStageMCMC(mcmc1=mcmc_coarse,
                                   mcmc2=mcmc_fine,
                                   len_chain1=len_coarse_chain,
                                   len_chain2=len_fine_chain,
                                   coarse_model=model_coarse,
                                   fine_over_coarse=fine_over_coarse_nsamp)

    fk_tempering = ssp.Tempering(model=model_fine,
                                 wastefree=False,
                                 exponents=[1.],
                                 move=two_stage_chain)

    my_temp_alg = SMC(fk=fk_tempering, N=nparticles, ESSrmin=1., 
                                verbose=False)
    my_temp_alg.run()
    my_temp_alg.X.shared["solver_nsamp"] = [Nmin] # (USEFUL ONLY FOR 2stage 1IS STEP RUN)
    my_temp_alg.setup_auxiliary_weights()
    my_temp_alg.resample_move()

    post_samp = my_temp_alg.X.theta['param']

    return post_samp