import numpy as np


class SamplerConfig:

    data_noise_var = 0.01
    ground_truth_par = 0.5*np.ones(4)
    prior_marg_var = 0.5
    n_particles = 100
    len_chain = 5
    ESS_rmin = 0.7
    ar_tol = 0.2
    Nmin = 100
