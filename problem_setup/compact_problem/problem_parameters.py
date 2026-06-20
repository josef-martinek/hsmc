import numpy as np


class SamplerConfig:

    problem = "compact"
    data_noise_var = 0.1

    param_dim = 5
    ground_truth_par = np.array([0.5, -0.5, 0.2, 0.8, -0.1])

    n_particles = 1000
    len_chain = 5
    ESS_rmin = 0.7
    ar_tol = 0.35
    Nmin = 32

    nsamp = 32
    fine_over_coarse_nsamp = None
    coarse_chain_scale = None

    prior_type = "uniform"
    prior_a = -2.0
    prior_b = 2.0
