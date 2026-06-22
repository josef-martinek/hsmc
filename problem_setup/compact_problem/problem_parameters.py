import numpy as np


class SamplerConfig:

    problem = "compact"
    data_noise_var = 0.4

    m = 10
    param_dim = 5  # Note: param_dim represents 'd'
    truncation_radius_R = 1.0
    base_variance = 1.6

    ground_truth_par = np.array([0.5, -0.5, 0.2, 0.8, -0.1])

    n_particles = 100
    len_chain = 10
    n_temp_steps = 10
    #ESS_rmin = 0.7
    #ar_tol = 0.35
    #Nmin = 32

    N_max = 0.02
    gamma_nsamp = 2
    gamma_temp = 1.5

    #fine_over_coarse_nsamp = None
    #coarse_chain_scale = None

    prior_type = "uniform"
    prior_a = -2.0
    prior_b = 2.0
