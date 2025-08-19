import numpy as np


class SamplerConfig:

    data_noise_var_seq = np.logspace(start=np.log10(0.1), stop=np.log10(0.0001), base=10, num=6)
    ground_truth_par = np.array([1.])
    prior_marg_var = 1.
    n_particles = 100
    len_chain = 2
    ESS_rmin = 0.9
    ar_tol = 0.35
    Nmin = 100
