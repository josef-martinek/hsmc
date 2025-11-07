import numpy as np


class SamplerConfig:

    problem = "wos"
    data_noise_var = 2.

    truth_rng = np.random.default_rng(seed=5)
    ground_truth_par = truth_rng.normal(scale=0.5, size=32)
    print("ground truth:", ground_truth_par)

    param_dim = 8
    prior_marg_var = 1.0
    n_particles = 100
    len_chain = 5
    ESS_rmin = 0.7
    ar_tol = 0.2
    Nmin = 0.5

    fine_over_coarse_nsamp = 20
    coarse_chain_scale = 2
