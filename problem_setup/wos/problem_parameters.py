import numpy as np


class SamplerConfig:

    data_noise_var = 0.05

    truth_rng = np.random.default_rng(seed=4)
    ground_truth_par = truth_rng.normal(scale=0.25, size=8)
    print("ground truth:", ground_truth_par)

    param_dim = 4
    prior_marg_var = 0.5
    n_particles = 100
    len_chain = 5
    ESS_rmin = 0.7
    ar_tol = 0.2
    Nmin = 2
