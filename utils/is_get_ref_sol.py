import numpy as np
from particles import smc_samplers as ssp
from particles.core import SMC
import core.mcmc as MCMC

def get_ref_sols(model_ex, len_chain=100, ESSrmin=0.9, N=1000, return_exponents=False):
    mcmc = MCMC.RWRegCal()
    move = ssp.AdaptiveMCMCSequence(mcmc=mcmc, adaptive=False, len_chain=len_chain)
    temp_ex = ssp.AdaptiveTempering(model=model_ex, wastefree=False, ESSrmin=ESSrmin, move=move)

    alg_ex = SMC(fk=temp_ex, N=N, ESSrmin=1., verbose=True)
    alg_ex.run()
    alg_ex.setup_auxiliary_weights()
    alg_ex.resample_move()
    post_samp_ex = alg_ex.X.theta['param']
    print("Exact reference sol. computed.")

    if return_exponents:
        exponents = np.array(alg_ex.X.shared['exponents'])
        return post_samp_ex, exponents
    return post_samp_ex

