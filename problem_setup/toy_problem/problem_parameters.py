import numpy as np


class SamplerConfig:

    #data_noise_var_seq = np.logspace(start=np.log10(0.01), stop=np.log10(0.0001), base=10, num=6)
    data_noise_var = 0.001
    param_dim = 12
    ground_truth_par = np.ones((param_dim,1))#/np.linalg.norm(np.ones((param_dim,1))) #Must be a scalar array (1D) or column vector (high dim)
    data_dim = 10
    prior_marg_var = 1.
    n_particles = 100
    len_chain = 33
    ESS_rmin = 0.7
    ar_tol = 0.23
    Nmin = 100
    fine_over_coarse_nsamp = 10
