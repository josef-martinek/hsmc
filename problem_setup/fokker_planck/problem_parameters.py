import numpy as np


class SamplerConfig:

    data_noise_var = 0.1
    ground_truth_par = np.array([-1., 1.])
    prior_marg_var = 1.
    n_particles = 100
    len_chain = 5
    ESS_rmin = 0.7
    ar_tol = 0.35
    Nmin = 100
