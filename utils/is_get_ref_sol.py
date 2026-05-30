import numpy as np
from particles import smc_samplers as ssp
from particles.core import SMC
import core.mcmc as MCMC
from problem_setup.wos.likelihood_model import IPModelExactHD

def get_ref_sols(prior, data_noise_var, data):

    model_ex = IPModelExactHD(prior=prior, 
                               data_noise_cov=data_noise_var, 
                               data=data)
    mcmc = MCMC.RWRegCal()
    move = ssp.AdaptiveMCMCSequence(mcmc=mcmc, adaptive=False, len_chain=100)
    temp_ex = ssp.AdaptiveTempering(model=model_ex, wastefree=False, ESSrmin=0.9, move=move)

    alg_ex = SMC(fk=temp_ex, N=1000, ESSrmin=1., verbose=True)
    alg_ex.run()
    alg_ex.setup_auxiliary_weights()
    alg_ex.resample_move()
    post_samp_ex = alg_ex.X.theta['param']
    print("Exact reference sol. computed.")

    return post_samp_ex

