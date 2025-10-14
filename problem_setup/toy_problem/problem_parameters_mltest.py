import numpy as np


class SamplerConfig:

    #data_noise_var_seq = np.logspace(start=np.log10(0.01), stop=np.log10(0.0001), base=10, num=6)
    data_noise_var = 0.001
    ground_truth_par = np.ones((32,1)) #Must be a scalar array (1D) or column vector (high dim)
    data_dim = 10
    prior_marg_var = 1.
    n_particles = 100
    len_chain = 33
    ESS_rmin = 0.7
    ar_tol = 0.25
    Nmin = 100
    fine_over_coarse_nsamp = 10