import numpy as np


class SamplerConfig:

    problem = "compact"
    data_noise_var = 0.1

    param_dim = 5
    ground_truth_par = np.array([0.5, -0.5, 0.2, 0.8, -0.1])

    prior_marg_var = 1.0
    n_particles = 20
    len_chain = 5
    ESS_rmin = 0.7
    ar_tol = 0.35
    Nmin = 32

    fine_over_coarse_nsamp = 32
    coarse_chain_scale = 1

    prior_type = "uniform"
    prior_a = -2.0
    prior_b = 2.0
