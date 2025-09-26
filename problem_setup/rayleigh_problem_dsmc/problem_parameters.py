import numpy as np


class SamplerConfig:

    data_noise_var= 0.02
    prior_mean = -0.5
    prior_marg_var = 0.5
    n_particles = 64
    len_chain = 4
    ESS_rmin = 0.7
    ar_tol = 0.4
    Nmin = 4
