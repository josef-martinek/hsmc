import numpy as np


class SamplerConfig:

    problem = "wos"
    data_noise_var = 3.2

    truth_rng = np.random.default_rng(seed=5)
    ground_truth_par = truth_rng.normal(scale=4.0, size=256) #+ 2.0 * np.ones(shape=(16,)) 
    #print("ground truth:", ground_truth_par)

    param_dim = 256
    prior_marg_var = 4.
    n_particles = 1000
    len_chain = 100
    ESS_rmin = 0.02
    ar_tol = 0.2
    Nmin = 2

    fine_over_coarse_nsamp = 32
    coarse_chain_scale = 1
