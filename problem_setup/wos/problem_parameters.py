import numpy as np


class SamplerConfig:

    problem = "wos"
    data_noise_var = 1.6

    param_dim = 2
    truth_rng = np.random.default_rng(seed=5)
    ground_truth_par = truth_rng.normal(scale=4.0, size=param_dim)
    #print("ground truth:", ground_truth_par)

    prior_marg_var = 4.
    n_particles = 100
    #len_chain = 100
    #ESS_rmin = 0.7
    #ar_tol = 0.2
    Nmin = 32

    fine_over_coarse_nsamp = 32
    coarse_chain_scale = 1
