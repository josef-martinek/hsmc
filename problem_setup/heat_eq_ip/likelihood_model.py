from problem_setup.likelihood_base import IPModel1D
import numpy as np
from scipy.stats import norm
import logging


class HeatEqModel:
    t = 0.1
    x_min, x_max = -1, 1
    n_bins = 20
    initial_sample_position = 0.
    dt = 0.001

    @staticmethod
    def param_to_diff(param):
        return np.exp(param)

    @classmethod
    def exact_sol(cls, param):
        """
        Computes the bin-integrated values of the Gaussian solution to the heat equation.

        Returns:
            piecewise_density (np.ndarray): Array of piecewise constant values (density per bin).
        """
        diff = cls.param_to_diff(param)
        sigma = np.sqrt(2 * diff * cls.t)
        bin_edges = np.linspace(cls.x_min, cls.x_max, cls.n_bins + 1)
        bin_widths = np.diff(bin_edges)
        cdf = norm(loc=0., scale=sigma).cdf

        piecewise_density = np.array([
            (cdf(b1) - cdf(b0)) / width
            for b0, b1, width in zip(bin_edges[:-1], bin_edges[1:], bin_widths)
        ])
        return piecewise_density
    
    @classmethod
    def euler_maruyama_binned(cls, param, nsamp):
        diff = cls.param_to_diff(param)
        x0 = np.full(nsamp, cls.initial_sample_position)
        tspan = np.linspace(0, cls.t, round(cls.t/cls.dt))
        x = x0.copy()
        for _ in range(1, len(tspan)):
            dW = np.sqrt(cls.dt) * np.random.normal(size=nsamp)
            x += np.sqrt(2 * diff) * dW  # No time/state dependence

        hist, bin_edges = np.histogram(x, bins=cls.n_bins, range=(cls.x_min, cls.x_max), density=False)
        bin_width = bin_edges[1] - bin_edges[0]
        hist_density = hist / (nsamp * bin_width)
        return hist_density

    @classmethod
    def euler_maruyama_binned_vectorized(cls, params, nsamp):
        """
        params: array of shape (n_params,)
        nsamp: number of samples per parameter
        returns: array of shape (n_params, n_bins)
        """
        diffs = cls.param_to_diff(params)  # shape (n_params,)
        n_params = len(diffs)

        tspan = np.linspace(0, cls.t, round(cls.t / cls.dt))
        n_steps = len(tspan)

        # Initial positions: shape (n_params, nsamp)
        x = np.full((n_params, nsamp), cls.initial_sample_position, dtype=np.float64)

        # Euler–Maruyama simulation (vectorized across params)
        for _ in range(1, n_steps):
            dW = np.sqrt(cls.dt) * np.random.normal(size=(n_params, nsamp))
            x += np.sqrt(2 * diffs[:, None]) * dW

        # Histogramming
        hist_array = np.empty((n_params, cls.n_bins), dtype=np.float64)
        bin_edges = np.linspace(cls.x_min, cls.x_max, cls.n_bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]

        for i in range(n_params):
            hist, _ = np.histogram(x[i], bins=bin_edges, density=False)
            hist_array[i] = hist / (nsamp * bin_width)

        return np.transpose(hist_array)


class IPModelExact(IPModel1D):
    def _evaluate_model(self, param, nsamp=None):
        if nsamp is not None:
            logging.warning("Number of samples was specified when getting exact solution.")
        return HeatEqModel.exact_sol(param)

    @property
    def parallel_eval(self):
        return False


class IPModelInexactFix(IPModel1D):
    def __init__(self, prior, data_noise_cov, data, nsamp):
        super().__init__(prior, data_noise_cov, data)
        self.nsamp = nsamp

    def _evaluate_model(self, param, nsamp=None):
        if nsamp is not None:
            logging.warning("Adaptive number of samples was specified when getting solution with fixed number of samples.")
        if self.parallel_eval:
            return HeatEqModel.euler_maruyama_binned_vectorized(param, self.nsamp)
        else:
            return HeatEqModel.euler_maruyama_binned(param=param, nsamp=self.nsamp)

    @property
    def parallel_eval(self):
        return True


class IPModelInexactAdapt(IPModel1D):
    def _evaluate_model(self, param, nsamp=None):
        if nsamp is None:
            logging.error("Number of samples must be specified for each model evaluation separately when using adaptive number of samples.")
        if self.parallel_eval:
            return HeatEqModel.euler_maruyama_binned_vectorized(param, nsamp)
        else:
            return HeatEqModel.euler_maruyama_binned(param, nsamp)

    @property
    def parallel_eval(self):
        return True
