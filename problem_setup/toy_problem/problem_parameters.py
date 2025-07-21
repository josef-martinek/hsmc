import numpy as np
from problem_setup.config_base import SamplerConfigBase
from abc import ABC, abstractmethod


class MRWConfig(SamplerConfigBase):

    def __init__(self):
        self._method = 'mrw'
        self._proposal_cov_type = 'iid'
        self._data_noise_var = np.float64(0.1)
        self._ground_truth_par = np.array([1])
        self._prior_marg_var = np.float64(1)
        self._proposal_marg_var = np.float64(1)
        self._n_steps = 50000
        self._burn_in = 5000
        self._init_state = np.array([4])
        self._thinning_step_default = None
        #self._solver_exact_setup = FwProblemExact()
        #self._solver_inference_setup = FwProblemExact()
        self._n_data = 1
        self._design = np.array([np.array([1]) for _ in range(self.n_data)])

    @property
    def solver_inference_setup(self):
        return self._solver_inference_setup
    

class DAConfig(MRWConfig):

    def __init__(self):
        super().__init__()
        self._method = 'mlda'
        self._n_steps = 50000
        self._burn_in = 5000
        self._subchain_lengths = [5]
        self._solver_fine_setup = FwProblemInexact(n_particles=100000)
        self._solver_coarse_setup = FwProblemInexact(n_particles=100)

    @property
    def solver_coarse_setup(self):
        return self._solver_coarse_setup
    
    @property
    def solver_fine_setup(self):
        return self._solver_fine_setup
    
    @property
    def subchain_lengths(self) -> list:
        return self._subchain_lengths
    

class PMConfig(MRWConfig):

    def __init__(self):
        super().__init__()
        self._method = 'pm'
        self._n_lkhd_samples = 1
        self._solver_inference_setup = FwProblemInexact(n_particles=1)
    
    @property
    def n_lkhd_samples(self) -> list:
        return self._n_lkhd_samples


class FwProblem(ABC):

    def __init__(self):
        #self._operator_dim = 10
        self._data_dim = 10
        self._parameter_dim = 1
        aux = np.ones((self._data_dim, self._parameter_dim))
        self._exact_fw_operator = aux/np.linalg.norm(aux)
        self._solver_data_noise_ratio = 100
        #self._solver_noise_var_constant = self._solver_data_noise_ratio*MRWConfig().data_noise_var
        self._solver_noise_var_constant = 1.
        self._fw_solve_repeat_M = 1

    @abstractmethod
    def evaluate(self, param):
        pass

    @property
    def fw_solve_repeat_M(self):
        return self._fw_solve_repeat_M

    @property
    def exact_fw_operator(self):
        return self._exact_fw_operator
    
    def get_data_dim(self):
        return self.exact_fw_operator.shape[0]
    
    @property
    def solver_noise_var(self) -> np.floating:
        return self._solver_noise_var
    
    @property
    def n_data(self):
        return self._n_data
    

class FwProblemExact(FwProblem):

    def __init__(self):
        super().__init__()
        self._solver_noise_var = 0

    def evaluate(self, param):
        return self.exact_fw_operator @ param


class FwProblemInexact(FwProblem):

    def __init__(self, n_particles):
        super().__init__()
        self._solver_noise_var = np.float64(self._solver_noise_var_constant * (1/n_particles))
        self._fw_solve_repeat_M = 1
        self._n_particles = n_particles

    # def evaluate(self, param):
    #     return self.exact_fw_operator @ param + np.random.normal(loc=0., scale=np.sqrt(self.solver_noise_var), size=self.exact_fw_operator.shape)
    
    def evaluate(self, param):
        return self.exact_fw_operator @ param + np.mean(np.random.normal(loc=0., scale=np.sqrt(self._solver_noise_var_constant), size=(self._n_particles, *self.exact_fw_operator.shape)), axis=0)
        
    @property
    def solver_noise_var(self):
        return self._solver_noise_var
    

class FwProblemInexactAdapt(FwProblem):

    def __init__(self):
        super().__init__()
        self._fw_solve_repeat_M = 1
    
    def evaluate(self, param, n_samp):
        return self.exact_fw_operator @ param + np.mean(np.random.normal(loc=0., scale=np.sqrt(self._solver_noise_var_constant), size=(n_samp, *self.exact_fw_operator.shape)), axis=0)
        
    @property
    def solver_noise_var(self):
        raise NotImplementedError
