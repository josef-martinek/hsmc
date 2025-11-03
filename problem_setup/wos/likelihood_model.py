from problem_setup.likelihood_base import IPModelHD
import numpy as np


class WOSHD:
    """
    Walk-On-Spheres class for the unit disk.
    - Only cosine coefficients are used: h(θ) = a0 + sum_{k=1}^s a_k cos(k θ) / k^q
    - Vectorized WOS over (params, points, paths) performed in batches over points.
    """

    # ----- simulation & model constants (tweak as needed) -----
    eps = 1e-2        # stopping threshold (distance to boundary)
    a0 = 1.          # constant boundary term (matches earlier example)
    q = 0.5           # decay exponent in boundary series
    R = 1.0           # unit disk

    # grid for evaluation (cell centres); we evaluate on full grid but only use interior points
    nx = 101
    ny = 101
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)

    @classmethod
    def grid_all(cls):
        """Return full grid array shape (nx*ny, 2)."""
        return np.array([(x, y) for x in cls.xs for y in cls.ys])

    @classmethod
    def mask_inside(cls):
        """Boolean mask of points inside the unit disk for grid_all()."""
        pts = cls.grid_all()
        r2 = pts[:, 0]**2 + pts[:, 1]**2
        return r2 < cls.R**2

    @classmethod
    def points_in(cls):
        """Return array of points inside disk, shape (n_points, 2)."""
        grid = cls.grid_all()
        mask = cls.mask_inside()
        return grid[mask]

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
    def wos(cls, params: np.ndarray, npaths: int, rng=None) -> np.ndarray:
        """
        Vectorized WOS for all interior points at once (no batching).
        
        params: (n_params, s) or (s,)
        npaths: number of MC paths per param per point
        rng: optional np.random.Generator
        returns: (n_points_total, n_params)
        """
        if rng is None:
            rng = np.random.default_rng()

        params = np.asarray(params)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        n_params = params.shape[0]

        # all interior points
        pts = cls.points_in()                        # (N_pts, 2)
        N_pts = pts.shape[0]

        # initialize positions for all paths
        # shape (n_params, N_pts, npaths, 2)
        P = np.repeat(pts[None, :, None, :], n_params, axis=0)
        P = np.repeat(P, npaths, axis=2)

        # mask for active paths
        active = np.ones((n_params, N_pts, npaths), dtype=bool)

        step_count = 0
        max_steps = 100000

        while np.any(active):
            step_count += 1
            x = P[..., 0]
            y = P[..., 1]
            r = np.hypot(x, y)

            dist = cls.R - r
            newly_done = (dist <= cls.eps) | (r >= cls.R)
            still_active = active & (~newly_done)
            if not np.any(still_active):
                active = active & (~newly_done)
                break

            # indices of active points
            idx = np.nonzero(still_active)
            d_vals = dist[idx]

            # sample random directions
            phi = rng.random(size=d_vals.shape)
            cos_phi = np.cos(2.0 * np.pi * phi)
            sin_phi = np.sin(2.0 * np.pi * phi)

            # update positions
            P[idx + (0,)] += d_vals * cos_phi
            P[idx + (1,)] += d_vals * sin_phi

            # update active mask
            x = P[..., 0]
            y = P[..., 1]
            r = np.hypot(x, y)
            dist = cls.R - r
            active = active & (dist > cls.eps) & (r < cls.R)

            if step_count > max_steps:
                raise RuntimeError("WOS: exceeded max steps for some path; increase eps or check geometry")

        # compute theta at boundary hits
        theta = np.arctan2(P[..., 1], P[..., 0])   # (n_params, N_pts, npaths)

        # evaluate boundary function
        hvals = cls.param_to_coeff(params, theta)  # (n_params, N_pts, npaths)

        # average over paths
        mean_h = hvals.mean(axis=2)                # (n_params, N_pts)

        # return (N_pts, n_params)
        return mean_h.T


    # ------------------------------------------------------------------
    # exact solution by harmonic extension (analytic)
    # params: (n_params, s)
    # returns: (n_points, n_params)
    # ------------------------------------------------------------------
    @classmethod
    def exact_sol(cls, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        n_params = params.shape[0]

        pts = cls.points_in()              # (N_pts, 2)
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