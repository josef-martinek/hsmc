import numpy as np


class SamplerConfig:

    problem = "wos"
    data_noise_var = 0.8

    truth_rng = np.random.default_rng(seed=5)
    ground_truth_par = - 0.2 * np.ones(shape=(32,)) + truth_rng.normal(scale=0.1, size=32)
    print("ground truth:", ground_truth_par)

    param_dim = 32
    prior_marg_var = 1.0
    n_particles = 100
    len_chain = np.array([2, 4])
    ESS_rmin = 0.7
    ar_tol = 0.2
    Nmin = 2

    fine_over_coarse_nsamp = 64
    coarse_chain_scale = 1
