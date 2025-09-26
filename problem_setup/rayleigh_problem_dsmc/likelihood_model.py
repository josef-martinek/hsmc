from problem_setup.likelihood_base import IPModel1D
import numpy as np
import logging
import numpy as np
import ray
from juliacall import Main as jl

# Initialize Ray once (if not already running)
ray.init(num_cpus=16, ignore_reinit_error=True)

# Worker function for a single Kn
@ray.remote(num_cpus=2)
def run_single_simulation(kn_val, nsamp):
    jl.seval("include(\"problem_setup/rayleigh_problem_dsmc/rayleigh.jl\")")
    ps, vs, params = jl.simulate(Kn=kn_val, Nsim=nsamp, Nmft=10, Ncell=32)
    mac = jl.stats(ps, vs, params)
    return np.array(mac.uy)  # extract velocity profile

def dsmc_simulation(kn, nsamp):
    """
    Run DSMC Rayleigh problem simulations for multiple Knudsen numbers in parallel.

    Parameters
    ----------
    kn : list or array-like
        Sequence of Knudsen numbers.
    nsamp : int
        Number of sampling particles (Nsim) to use.

    Returns
    -------
    np.ndarray
        Array of shape (len(uy), len(kn)) with simulation results.
        Each column corresponds to uy for a given Kn.
    """
    # Launch tasks in parallel
    futures = [run_single_simulation.remote(k, nsamp) for k in kn]

    # Gather results
    results = ray.get(futures)

    # Stack into (len(uy), len(kn))
    results_matrix = np.column_stack(results)

    return results_matrix


class IPModelInexactFix(IPModel1D):
    def __init__(self, prior, data_noise_cov, data, nsamp):
        super().__init__(prior, data_noise_cov, data)
        self._nsamp = round(nsamp)

    def _evaluate_model(self, param):
        if not self.parallel_eval:
            logging.warning("Rayleigh problem only supports parallel evaluation.")
        return dsmc_simulation(kn=self.param_to_kn(param),
                               nsamp=self.nsamp)
    
    def param_to_kn(self, param):
        return 10.**param

    @property
    def parallel_eval(self):
        return True

    @property
    def nsamp(self):
        return self._nsamp

    @nsamp.setter
    def nsamp(self, value):
        raise AttributeError("nsamp is read only in model with fixed number of samples.")


class IPModelInexactAdapt(IPModel1D):
    def __init__(self, prior, data_noise_cov, data):
        super().__init__(prior, data_noise_cov, data)
        self._nsamp = None

    def _evaluate_model(self, param):
        if not self.parallel_eval:
            logging.warning("Rayleigh problem only supports parallel evaluation.")
        return dsmc_simulation(kn=self.param_to_kn(param),
                               nsamp=self.nsamp)
    
    def param_to_kn(self, param):
        return 10.**param

    @property
    def parallel_eval(self):
        return True

    @property
    def nsamp(self):
        if self._nsamp is None:
            raise AttributeError("nsamp not specified in model with adaptive number of samples")
        return self._nsamp

    @nsamp.setter
    def nsamp(self, value):
        self._nsamp = round(value)
