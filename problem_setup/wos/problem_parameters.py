import numpy as np


class SamplerConfig:

    data_noise_var = 0.01

    truth_rng = np.random.default_rng(seed=4)
    ground_truth_par = truth_rng.normal(scale=0.25, size=32)
    print("ground truth:", ground_truth_par)

    param_dim = 2
    prior_marg_var = 0.5
    n_particles = 100
    len_chain = 5
    ESS_rmin = 0.7
    ar_tol = 0.2
    Nmin = 2
