import numpy as np


class SamplerConfig:

    data_noise_var_seq = np.logspace(start=np.log10(0.1), stop=np.log10(0.000001), base=10, num=10)
    ground_truth_par = np.array([1.])
    prior_marg_var = 1.
    n_particles = 100
    len_chain = 10
    ESS_rmin = 0.3