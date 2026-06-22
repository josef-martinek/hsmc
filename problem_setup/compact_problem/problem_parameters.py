import numpy as np


class SamplerConfig:

    problem = "compact"
    data_noise_var = 0.1

    m = 10
    param_dim = 5  # Note: param_dim represents 'd'
    truncation_radius_R = 1.0
    base_variance = 2.0

    ground_truth_par = np.array([0.5, -0.5, 0.2, 0.8, -0.1])

    n_particles = 1000
    len_chain = 5
    n_temp_steps = 10
    ESS_rmin = 0.7
    ar_tol = 0.35
    Nmin = 32

    N_max = 64.0
    gamma_nsamp = 1.25
    gamma_temp = 1.5

    fine_over_coarse_nsamp = None
    coarse_chain_scale = None

    prior_type = "uniform"
    prior_a = -2.0
    prior_b = 2.0
