import numpy as np
from scipy.linalg import solve_banded


def solve_fp(x, D_field, p0, dt, t_final, theta=0.5):
    """
    Fast, banded Crank–Nicolson solver for ∂_t p = ∂_x^2( D(x) p ), Dirichlet p=0.
    This function mirrors exactly the original discretisation in solve_fd_Dp_xx
    but uses scipy.solve_banded for the tridiagonal solve.

    Parameters
    ----------
    x : (N,) array
        cell centers
    D_field : (N,) array
        D(x) on cell centers
    p0 : (N,) array
        initial condition
    dt : float
    t_final : float
    theta : float (0.5 CN, 1.0 implicit Euler)

    Returns
    -------
    u : (N,) array
        solution at final time
    """
    x = np.asarray(x)
    N = len(x)
    dx = x[1] - x[0]

    # number of time steps (adjust dt so it divides t_final)
    n_steps = int(np.ceil(t_final / dt))
    dt = t_final / n_steps

    # Build operator A approximating ∂_x^2 (D u)
    # Follow exactly the same indexing as in your reference solver.
    lower_full = D_field[:-1] / (dx * dx)   # length N-1, corresponds to A[i,i-1] = D_{i-1}/dx^2
    upper_full = D_field[1:]  / (dx * dx)   # length N-1, corresponds to A[i,i+1] = D_{i+1}/dx^2
    diag_full  = np.zeros(N, dtype=float)   # length N
    diag_full[1:-1] = -2.0 * D_field[1:-1] / (dx * dx)

    # LHS = I - theta * dt * A  (a_lhs, b_lhs, c_lhs follow original lengths)
    a_lhs = -theta * dt * lower_full.copy()   # length N-1 (subdiagonal)
    b_lhs = 1.0 - theta * dt * diag_full.copy() # length N (diagonal)
    c_lhs = -theta * dt * upper_full.copy()   # length N-1 (superdiagonal)

    # enforce Dirichlet BC rows (make first and last rows identity)
    b_lhs[0] = 1.0
    a_lhs[0:1] = 0.0
    c_lhs[0:1] = 0.0
    b_lhs[-1] = 1.0
    a_lhs[-1:] = 0.0
    c_lhs[-1:] = 0.0

    # convert to SciPy banded form (ab shape (3, N) with (1,1) super/subdiags)
    # ab[0,1:] = superdiag (c_lhs)
    # ab[1,:]  = diag (b_lhs)
    # ab[2,:-1]= subdiag (a_lhs)
    ab = np.zeros((3, N), dtype=float)
    ab[0, 1:] = c_lhs        # upper diag shifted right
    ab[1, :]  = b_lhs
    ab[2, :-1] = a_lhs       # lower diag shifted left

    # initial condition
    u = p0.copy()

    # time stepping
    for _ in range(n_steps):
        # Compute Au using same discretisation as used to build A
        Au = np.zeros_like(u)
        # interior nodes: note we use exactly the same slices as the reference solver
        # lower_full[0:N-2] multiplies u[0:-2], upper_full[1:N-1] multiplies u[2:]
        if N > 2:
            Au[1:-1] = lower_full[0:N-2] * u[0:-2] + diag_full[1:-1] * u[1:-1] + upper_full[1:N-1] * u[2:]
        # boundaries (Dirichlet => Au[0]=Au[-1]=0)
        Au[0] = 0.0
        Au[-1] = 0.0

        # RHS = (I + (1-theta)*dt*A) u
        rhs = u + (1.0 - theta) * dt * Au
        rhs[0] = 0.0
        rhs[-1] = 0.0

        # Solve banded system
        u = solve_banded((1, 1), ab, rhs)

    return u