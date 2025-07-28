import numpy as np


class SamplerConfig:

    data_noise_var = 0.001
    ground_truth_par = np.array([np.log(0.5)])  #in log space! (true diffusion is exp(ground_truth))
    prior_marg_var = 1.
    n_data = 1
    n_samples = 10000
    n_particles = 100
    len_chain = 10
