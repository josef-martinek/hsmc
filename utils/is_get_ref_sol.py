import numpy as np
from particles import smc_samplers as ssp
from particles.core import SMC
import core.mcmc as MCMC

def get_ref_sols(model_ex):
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

