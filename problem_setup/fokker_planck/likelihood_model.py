from problem_setup.likelihood_base import IPModel1D
import numpy as np
from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm


class FPModelHD:
    # Implements model for inference of the diffusion coefficient of 1D Fokker-Planck equation
    # with zero drift. The coefficient is given as a fin. dim. lognormal RF represented by
    # a series. As the solver we take the corresponding EM discretisation of the Ito SDE.
    x_min, x_max = -1.0, 1.0 #Part of domain where we take measurements.
    n_bins = 50
    t = 0.1
    dt = 1e-4
    # Gaussian initial distribution
    x0 = 0.0
    sigma0 = 0.05

    #fipy (finite volumes) setup
    x_min_fv, x_max_fv = -4.0, 4.0
    nx_fv = 1000
    dx_fv = (x_max_fv - x_min_fv) / nx_fv
    mesh_fv = Grid1D(dx=dx_fv, nx=nx_fv)
    x_fv = mesh_fv.cellCenters[0].value + x_min_fv
    norm_const = 1.0 / (np.sqrt(2 * np.pi) * sigma0)
    p0 = np.exp(-((x_fv - x0) ** 2) / (2 * sigma0**2)) * norm_const
    dt_fv = 1e-4
    n_steps_fv = int(t / dt_fv)

    # New basis / field parameters
    s = 5        # number of cosine modes (s in your sum) #TODO maybe just determine adaptively
    q = 1.0       # decay exponent for 1/j^q
    a0 = 0.0      # constant offset a0 in exponent

    @staticmethod
    def param_to_diff(params: np.ndarray, x: np.ndarray):
        """
        Evaluate D(x) = exp(a0 + sum_j omega_j * (1/j^q) * cos(2π j x))
        exactly at the given x_positions (no interpolation).
        - params: array-like, shape (n_params, s)
        - x: array-like, shape (n_params, nsamp)
                       (each row gives the positions for that parameter set)
        Returns:
        - D: ndarray shape (n_params, nsamp)
        """

        if x.ndim != 2:
            raise ValueError("x_positions must have shape (n_params, nsamp)")

        # j indices: 1..s
        j = np.arange(1, FPModelHD.s + 1)
        scales = 1.0 / (j ** FPModelHD.q)    # shape (s,)

        # Evaluate cos(2π j x).
        # Build array of shape (s, n_params, nsamp): cos_j[j,i,k] = cos(2π * j * x[i,k])
        # Use broadcasting: j[:,None,None] * x[None,:,:]
        cos_j = np.cos(2.0 * np.pi * j[:, None, None] * x[None, :, :])  # (s, n_params, nsamp)

        # scale the params by scales: params_scaled shape (n_params, s)
        params_scaled = params * scales[None, :]  # (n_params, s)

        # sum_j params_scaled[i,j] * cos_j[j,i,k]  -> result shape (n_params, nsamp)
        # einsum over j: 'ij,jik->ik'
        m = FPModelHD.a0 + np.einsum('ij,jik->ik', params_scaled, cos_j)

        D = np.exp(m)  # (n_params, nsamp)
        return D

    @classmethod
    def euler_maruyama_binned_vectorized(cls, params, nsamp):
        """
        params: array-like of shape (n_params, s) # TODO check
        nsamp: number of samples per parameter
        returns: array of shape (n_bins, n_params) #TODO check
        """
        params = np.asarray(params, dtype=np.float64)
        n_params = params.shape[0]

        tspan = np.linspace(0, cls.t, int(round(cls.t / cls.dt)))
        n_steps = len(tspan)

        # Initial positions: shape (n_params, nsamp)
        x = np.random.normal(loc=cls.x0, scale=cls.sigma0, size=(n_params, nsamp))

        # Euler–Maruyama simulation (vectorized across params)
        for _ in range(1, n_steps):
            dW = np.sqrt(cls.dt) * np.random.normal(size=(n_params, nsamp))
            # Evaluate spatially varying diffusivity at current positions
            local_diff = cls.param_to_diff(params, x)  # shape (n_params, nsamp)
            # Update
            x += np.sqrt(2.0 * local_diff) * dW

        # Histogramming
        hist_array = np.empty((n_params, cls.n_bins), dtype=np.float64)
        bin_edges = np.linspace(cls.x_min, cls.x_max, cls.n_bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]

        for i in range(n_params):
            hist, _ = np.histogram(x[i], bins=bin_edges, density=False)
            hist_array[i] = hist / (nsamp * bin_width)

        # return shape (n_bins, n_params) to match your original return
        return np.transpose(hist_array)
    
    @classmethod
    def fv_solution(cls, params):
        D_field = cls.param_to_diff(params=params.reshape(1, -1),
                                    x=cls.x_fv.reshape(1, -1)).flatten()
        D = CellVariable(mesh=cls.mesh_fv, value=D_field)
        p_fv = CellVariable(mesh=cls.mesh_fv, value=cls.p0, name="p")
        eq = TransientTerm() == DiffusionTerm(coeff=D)
        for step in range(cls.n_steps_fv):
            eq.solve(var=p_fv, dt=cls.dt_fv)
        sol = p_fv.value.copy() # TODO check what this is doing
        sol = cls.postprocess_fv_sol(sol.reshape(-1,1)).flatten()
        # TODO sanity check
        return sol
    
    @classmethod
    def postprocess_fv_sol(cls, sol):
        """
        Project the fine FiPy solution(s) onto a coarser mesh by truncating and bin-averaging.

        Parameters
        ----------
        sol : ndarray
            Fine solution(s) on the FiPy mesh.
            Shape (nx_fv, n_params) for multiple.

        Returns
        -------
        sol_coarse : ndarray
            Coarse, binned solution(s) over [x_min, x_max].
            Shape (n_bins, n_params).
        """
        n_bins_fv, n_params = sol.shape
        if sol[0]>1e-8 or sol[-1]>1e-8:
            print(f"The boundary solution values are large, namely sol(x_min)={sol[0]} and sol(x_max)={sol[-1]}")

        # --- Step 1: truncate fine grid to [x_min, x_max] ---
        mask = (cls.x_fv >= cls.x_min) & (cls.x_fv <= cls.x_max)
        x_trunc = cls.x_fv[mask]
        sol_trunc = sol[mask, :]

        # --- Step 2: define coarse grid ---
        # uniform bins in [x_min, x_max]
        x_edges = np.linspace(cls.x_min, cls.x_max, cls.n_bins + 1)
        n_bins_out = cls.n_bins

        # --- Step 3: bin-average the fine solution over coarse bins ---
        sol_coarse = np.zeros((n_bins_out, n_params))
        for i in range(n_bins_out):
            in_bin = (x_trunc >= x_edges[i]) & (x_trunc < x_edges[i + 1])
            if np.any(in_bin):
                sol_coarse[i, :] = np.mean(sol_trunc[in_bin, :], axis=0)
            else:
                raise ValueError("Projecting solution failed.")

        return sol_coarse



class IPModelExact(IPModel1D):
    def _evaluate_model(self, param):
        return HeatEqModel.exact_sol(param)

    @property
    def parallel_eval(self):
        return False

    @property
    def nsamp(self):
        raise AttributeError("nsamp not used in exact model.")

    @nsamp.setter
    def nsamp(self, value):
        raise AttributeError("nsamp not used in exact model.")


class IPModelInexactFix(IPModel1D):
    def __init__(self, prior, data_noise_cov, data, nsamp):
        super().__init__(prior, data_noise_cov, data)
        self._nsamp = round(nsamp)

    def _evaluate_model(self, param):
        if self.parallel_eval:
            return HeatEqModel.euler_maruyama_binned_vectorized(params=param,
                                                                nsamp=self.nsamp)
        else:
            return HeatEqModel.euler_maruyama_binned(param=param,
                                                     nsamp=self.nsamp)

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
        if self.parallel_eval:
            return HeatEqModel.euler_maruyama_binned_vectorized(params=param,
                                                                nsamp=self.nsamp)
        else:
            return HeatEqModel.euler_maruyama_binned(param=param,
                                                     nsamp=self.nsamp)

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
