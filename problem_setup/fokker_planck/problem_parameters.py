import numpy as np


class SamplerConfig:

    data_noise_var_seq = np.logspace(start=np.log10(0.1), stop=np.log10(0.005), base=10, num=2)
    ground_truth_par = np.array([np.log(0.5)])  #in log space! (true diffusion is exp(ground_truth))
    prior_marg_var = 1.
    n_particles = 100
    len_chain = 10
    ESS_rmin = 0.7
    ar_tol = 0.35
    Nmin = 100
