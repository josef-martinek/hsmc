import numpy as np
from experiments.complex_compact_gaussian_prior_smc import ComplexCompactGaussianPriorSMCExperiment

class GaussianPriorDiscreteExperiment(ComplexCompactGaussianPriorSMCExperiment):
    @staticmethod
    def get_adaptive_schedule(nmax, gamma_nsamp, n_temp_steps, nsamp_min, shape_p=1.0):
        import numpy as np
        nsamp_sched = np.zeros(n_temp_steps + 1)
        
        for t in range(n_temp_steps + 1):
            # 1. Calculate the ideal smooth geometric curve value with shape_p
            smooth_val = nsamp_min * ( (nmax / nsamp_min) ** ( (t / n_temp_steps)**shape_p ) )
            
            # 2. Find the corresponding continuous power of gamma_nsamp
            k_continuous = np.log(smooth_val / nsamp_min) / np.log(gamma_nsamp)
            
            # 3. Snap to the nearest integer power
            k_snapped = np.round(k_continuous)
            
            # 4. Calculate the snapped nsamp
            snapped_val = nsamp_min * (gamma_nsamp ** k_snapped)
            
            # Ensure it doesn't exceed nmax (due to floating point inaccuracies)
            nsamp_sched[t] = min(snapped_val, nmax)
            
        return nsamp_sched
