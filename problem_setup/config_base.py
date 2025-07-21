from abc import ABC, abstractmethod
import numpy as np
import logging


class SamplerConfigBase(ABC):

    @property
    def method(self) -> str:
        return self._method

    @property
    def proposal_cov_type(self) -> str:
        return self._proposal_cov_type

    @property
    def data_noise_var(self) -> np.floating:
        return self._data_noise_var

    @property
    def ground_truth_par(self) -> np.ndarray:
        return self._ground_truth_par.astype(np.float64)

    @property
    def prior_marg_var(self) -> np.floating:
        return self._prior_marg_var

    @property
    def proposal_marg_var(self) -> np.floating:
        return self._proposal_marg_var

    @property
    def n_steps(self) -> np.integer:
        return self._n_steps

    @property
    def burn_in(self) -> np.integer:
        return self._burn_in

    @property
    def init_state(self) -> np.ndarray:
        return self._init_state.astype(np.float64)
    
    @property
    def n_data(self) -> np.integer:
        return self._n_data
    
    @property
    def thinning_step_default(self):
        return self._thinning_step_default
    
    @property
    def solver_exact_setup(self):
        return self._solver_exact_setup
    
    @property
    def design(self):
        return self._design