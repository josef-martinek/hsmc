from problem_setup.likelihood_base import IPModelHD
import numpy as np


class WOSHD:
    """
    Walk-On-Spheres class for the unit disk.
    - Only cosine coefficients are used: h(θ) = a0 + sum_{k=1}^s a_k cos(k θ) / k^q
    - Vectorized WOS over (params, points, paths) performed in batches over points.
    """

    # ----- simulation & model constants (tweak as needed) -----
    eps = 1e-3        # stopping threshold (distance to boundary)
    a0 = 0.2          # constant boundary term (matches earlier example)
    q = 1.1           # decay exponent in boundary series
    R = 1.0           # unit disk
    r = 0.5           # inner boundary

    # grid for evaluation (cell centres); we evaluate on full grid but only use interior points
    nx = 1000
    ny = 1000
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    nr = 3
    nphi = 10

    @classmethod
    def grid_all(cls):
        """Return full grid array shape (nx*ny, 2)."""
        return np.array([(x, y) for x in cls.xs for y in cls.ys])

    @classmethod
    def mask_inside(cls):
        """Boolean mask of points inside the unit disk for grid_all()."""
        pts = cls.grid_all()
        r2 = pts[:, 0]**2 + pts[:, 1]**2
        return (r2 < cls.R**2) & (r2 >= cls.r**2)

    @classmethod
    def points_in(cls):
        rs = np.linspace(cls.r, cls.R, cls.nr, endpoint=True)
        thetas = np.linspace(0, 2*np.pi, cls.nphi, endpoint=False)

        pts = [(r * np.cos(t), r * np.sin(t)) for r in rs for t in thetas]
        return np.array(pts)



    @classmethod
    def n_points(cls):
        return cls.points_in().shape[0]

    # ------------------------------------------------------------------
    # param_to_coeff: evaluate boundary function h(θ) for given params and theta
    # params: (n_params, s)
    # theta:  (n_params, n_points_batch, n_paths)
    # returns: (n_params, n_points_batch, n_paths)
    # ------------------------------------------------------------------
    @classmethod
    def param_to_coeff(cls, params: np.ndarray, theta: np.ndarray) -> np.ndarray:
        params = np.asarray(params)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        n_params = params.shape[0]
        s = params.shape[1]

        # theta shape may be (n_params, n_points_batch, n_paths) or broadcastable to that
        # We want cos(k*theta) with k=1..s and include decay factor 1/k^q
        ks = np.arange(1, s + 1)
        decay = (1.0 / (ks ** cls.q))[None, None, None, :]  # shape (1,1,1,s)

        # ensure theta has shape (n_params, n_points, n_paths)
        theta = np.asarray(theta)
        # broadcast theta to (n_params, n_points, n_paths) if needed
        # If theta lacks param axis, expand:
        if theta.ndim == 2:
            # (n_points, n_paths) -> (1, n_points, n_paths)
            theta = theta[None, ...]
        elif theta.ndim == 1:
            # (n_paths,) -> (1,1,n_paths)
            theta = theta[None, None, :]

        # compute cos(k*theta) -> shape (n_params, n_points, n_paths, s)
        cos_terms = np.cos(theta[..., None] * ks)

        # params has shape (n_params, s) -> broadcast to (n_params, 1, 1, s)
        params4 = params[:, None, None, :]

        # multiply and sum over s
        val = np.sum(params4 * cos_terms * decay, axis=-1)  # (n_params, n_points, n_paths)

        # add constant offset a0
        if cls.a0 != 0.0:
            val = val + cls.a0

        return val

    # ------------------------------------------------------------------
    # wos: perform WOS for many evaluation points in batches
    # params: (n_params, s)
    # npaths: scalar int, number of WOS paths per (param, point)
    # batch_points: how many grid points to process in one batch (memory control)
    # returns: (n_points_total, n_params)
    # ------------------------------------------------------------------

    @classmethod
    def wos(cls, params: np.ndarray, npaths: float, batch_points: int = 100) -> np.ndarray:
        """
        Batched Walk-On-Spheres for annulus (r < |x| < R).

        params: (n_params, s) or (s,)
        npaths: number of paths per (param, point);
                if 0 < npaths < 1, interpreted as fraction of points sampled.
        batch_points: number of grid points processed simultaneously
        returns: (N_pts, n_params)
        """

        params = np.asarray(params)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        n_params = params.shape[0]

        pts = cls.points_in()  # (N_pts, 2)
        N_pts = pts.shape[0]

        out = np.zeros((N_pts, n_params))
        max_steps = 100000

        # Determine sampling strategy
        if npaths < 1.0:
            sampled_idx = np.random.choice(N_pts, size=int(round(N_pts * npaths)), replace=False)
            npaths_int = 1
        else:
            sampled_idx = np.arange(N_pts)
            npaths_int = int(npaths)

        for ip in range(n_params):
            p = params[ip]  # shape (s,)

            # Process sampled points in batches
            for start in range(0, len(sampled_idx), batch_points):
                sel = sampled_idx[start:start + batch_points]
                batch_pts = pts[sel]  # (m, 2)
                m = len(sel)

                # Init positions for all paths at once: (m, npaths_int, 2)
                P = np.zeros((m, npaths_int, 2))
                P[:, :, 0] = batch_pts[:, 0, None]
                P[:, :, 1] = batch_pts[:, 1, None]

                active = np.ones((m, npaths_int), dtype=bool)
                step_count = 0

                while np.any(active):
                    step_count += 1

                    x = P[..., 0]
                    y = P[..., 1]
                    r = np.hypot(x, y)
                    dist = cls.R - r

                    newly_done = (dist <= cls.eps) | (r >= cls.R)
                    active &= ~newly_done

                    if not np.any(active):
                        break

                    idx = np.where(active)
                    d_vals = dist[idx]

                    phi = np.random.random(size=d_vals.shape)
                    cos_phi = np.cos(2.0 * np.pi * phi)
                    sin_phi = np.sin(2.0 * np.pi * phi)

                    P[idx + (0,)] += d_vals * cos_phi
                    P[idx + (1,)] += d_vals * sin_phi

                    x = P[..., 0]
                    y = P[..., 1]
                    r = np.hypot(x, y)
                    dist = cls.R - r
                    active &= (dist > cls.eps) & (r < cls.R)

                    if step_count > max_steps:
                        raise RuntimeError("WOS: exceeded max steps; increase eps or check geometry")

                # After WOS loop, compute boundary theta for all paths
                theta = np.arctan2(P[..., 1], P[..., 0])  # (m, npaths_int)
                val = cls.param_to_coeff(p, theta[None, :, :])  # -> (1, m, npaths_int)
                mean_val = val.mean(axis=-1).squeeze()          # (m,)

                out[sel, ip] = mean_val

        return out


    # ------------------------------------------------------------------
    # exact solution by harmonic extension (analytic)
    # params: (n_params, s)
    # returns: (n_points, n_params)
    # ------------------------------------------------------------------
    @classmethod
    def exact_sol(cls, params: np.ndarray) -> np.ndarray:
        pts = cls.points_in()              # (N_pts, 2)
        return cls.exact_sol_pts(params, pts)
    
    @classmethod
    def exact_sol_full(cls, params: np.ndarray) -> np.ndarray:
        grid = cls.grid_all()
        r2 = grid[:, 0]**2 + grid[:, 1]**2
        pts_inside = grid[r2<cls.R]
        return cls.exact_sol_pts(params, pts_inside)

    @classmethod
    def exact_sol_pts(cls, params: np.ndarray, pts:np.ndarray) -> np.ndarray:
        params = np.asarray(params)
        if params.ndim == 1:
            params = params.reshape(1, -1)

        x = pts[:, 0]; y = pts[:, 1]
        r = np.hypot(x, y)                 # (N_pts,)
        theta = np.arctan2(y, x)           # (N_pts,)

        ks = np.arange(1, params.shape[1] + 1)
        # compute r^k * cos(k theta) -> (N_pts, s)
        cosk = np.cos(theta[:, None] * ks) * (r[:, None] ** ks)
        decay = 1.0 / (ks ** cls.q)
        # cosk * decay -> (N_pts, s)
        coef_matrix = cosk * decay[None, :]

        # params.T -> (s, n_params). multiply -> (N_pts, n_params)
        u = coef_matrix @ params.T  # (N_pts, n_params)
        if cls.a0 != 0.0:
            u = u + cls.a0
        return u
    

class IPModelExactHD(IPModelHD):
    def _evaluate_model(self, param):
        return WOSHD.exact_sol(param)

    @property
    def parallel_eval(self):
        return True

    @property
    def nsamp(self):
        raise AttributeError("nsamp not used in exact model.")

    @nsamp.setter
    def nsamp(self, value):
        raise AttributeError("nsamp not used in exact model.")


class IPModelInexactFixHD(IPModelHD):
    def __init__(self, prior, data_noise_cov, data, nsamp):
        super().__init__(prior, data_noise_cov, data)
        self._nsamp = nsamp

    def _evaluate_model(self, param):
        return WOSHD.wos(params=param,
                         npaths=self.nsamp)

    @property
    def parallel_eval(self):
        return True

    @property
    def nsamp(self):
        return self._nsamp

    @nsamp.setter
    def nsamp(self, value):
        raise AttributeError("nsamp is read only in model with fixed number of samples.")

    def cost_normalising_const(self):
        const = WOSHD.points_in().shape[0]
        return const


class IPModelInexactAdaptHD(IPModelHD):
    def __init__(self, prior, data_noise_cov, data):
        super().__init__(prior, data_noise_cov, data)
        self._nsamp = None

    def _evaluate_model(self, param):
        if self.parallel_eval:
            return WOSHD.wos(params=param,
                             npaths=self.nsamp)

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
        self._nsamp = value

    def cost_normalising_const(self):
        const = WOSHD.points_in().shape[0]
        return const
