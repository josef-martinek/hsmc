import numpy as np
import scipy.stats as stats
from particles import distributions as dists
from particles import smc_samplers as ssp
from core.samplers import TempAdaptNsampNoRew
from particles.core import SMC
import core.mcmc as MCMC
from problem_setup.wos.likelihood_model import IPModelExactHD, IPModelInexactFixHD, IPModelInexactAdaptHD

def get_ref_sols(prior, data_noise_var, data, nsamp_fix, WOS):

    model_ex = IPModelExactHD(prior=prior, 
                               data_noise_cov=data_noise_var, 
                               data=data)
    model_wos = IPModelInexactFixHD(prior=prior, 
                                      data_noise_cov=data_noise_var, 
                                      data=data,
                                      nsamp=nsamp_fix)
    mcmc = MCMC.RWRegCal()
    move = ssp.AdaptiveMCMCSequence(mcmc=mcmc, adaptive=False, len_chain=100)
    temp_ex = ssp.AdaptiveTempering(model=model_ex, wastefree=False, ESSrmin=0.9, move=move)
    temp_wos = ssp.AdaptiveTempering(model=model_wos, wastefree=False, ESSrmin=0.9, move=move)

    alg_ex = SMC(fk=temp_ex, N=1000, ESSrmin=1., verbose=True)
    alg_ex.run()
    alg_ex.setup_auxiliary_weights()
    alg_ex.resample_move()
    post_samp_ex = alg_ex.X.theta['param']
    print("Exact reference sol. computed.")

    if WOS:
        alg_wos = SMC(fk=temp_wos, N=1000, ESSrmin=1., verbose=True)
        alg_wos.run()
        alg_wos.setup_auxiliary_weights()
        alg_wos.resample_move()
        post_samp_wos = alg_wos.X.theta['param']
        print("WOS reference sol. computed")

        return post_samp_ex, post_samp_wos

    else:
        print("No WOS reference solution.")
        return post_samp_ex, np.zeros_like(post_samp_ex)
