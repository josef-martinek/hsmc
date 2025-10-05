import numpy as np


class SamplerConfig:

    data_noise_var_seq = np.logspace(start=np.log10(0.01), stop=np.log10(0.0001), base=10, num=6)
    ground_truth_par = np.array([[1.], [1.]]) #Must be a scalar array (1D) or column vector (high dim)
    data_dim = 10
    prior_marg_var = 2.
    n_particles = 1000
    len_chain = 20
    ESS_rmin = 0.9
    ar_tol = 0.35
    Nmin = 100
