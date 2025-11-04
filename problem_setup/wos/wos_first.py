#!/usr/bin/env python3
"""
Walk-On-Spheres (WOS) solver for Laplace on the unit disk with Fourier-series
Dirichlet boundary. Compares Monte-Carlo WOS to analytic harmonic extension.

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Model / parameters
# -----------------------------
np.random.seed(1)

s = 6               # number of Fourier modes
q = 1.5             # decay exponent (smoothness)
a0 = 0.5            # constant term
# True Fourier coefficients (you could randomize or set)
omega_cos = np.array([1.0, -0.5, 0.3, 0.0, 0.0, 0.0])   # length s
omega_sin = np.array([0.2, 0.0, -0.2, 0.0, 0.0, 0.0])   # length s

# WOS parameters
eps = 1e-1          # shell width: stop when distance to boundary <= eps
n_paths = 2      # number of WOS paths per evaluation point (increase for accuracy)
max_steps = 10000   # safety cap for WOS steps per path

# grid to evaluate solution
nx = 51
ny = 51
X = np.linspace(-1.0, 1.0, nx)
Y = np.linspace(-1.0, 1.0, ny)
xx, yy = np.meshgrid(X, Y)
points = np.stack([xx.ravel(), yy.ravel()], axis=1)

# mask to keep only points inside unit disk
r2 = points[:,0]**2 + points[:,1]**2
inside_mask = r2 < 1.0
pts_in = points[inside_mask]

# -----------------------------
# Boundary function h(θ) and analytic solution
# -----------------------------
def h_boundary(theta, a0, omega_cos, omega_sin, q):
    """
    Evaluate boundary function h at angle(s) theta.
    theta: shape (...) in radians
    returns shape (...)
    """
    theta = np.asarray(theta)
    out = a0 + 0.0*theta
    n = np.arange(1, len(omega_cos)+1)
    # broadcast: (n,) with theta shape
    for j in range(len(n)):
        out = out + (omega_cos[j] * np.cos(n[j]*theta) + omega_sin[j] * np.sin(n[j]*theta)) / (n[j]**q)
    return out

def analytic_solution(points, a0, omega_cos, omega_sin, q):
    """
    Compute analytic harmonic extension inside unit disk at cartesian points.
    points: (M,2)
    returns u (M,)
    """
    x = points[:,0]; y = points[:,1]
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y, x)
    out = a0 + 0.0*r
    n = np.arange(1, len(omega_cos)+1)
    for j in range(len(n)):
        out += (r**n[j]) * (omega_cos[j] * np.cos(n[j]*theta) + omega_sin[j] * np.sin(n[j]*theta)) / (n[j]**q)
    return out

# -----------------------------
# Walk-On-Spheres implementation
# -----------------------------
def wos_single_point(x0, n_paths, eps=1e-3, max_steps=10000):
    """
    WOS for a single starting point x0 (array-like length 2).
    Returns estimated u(x0) by averaging boundary hits.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    estimates = np.empty(n_paths, dtype=np.float64)
    for p in range(n_paths):
        x = x0.copy()
        # perform WOS until within eps of boundary
        for step in range(max_steps):
            r = np.hypot(x[0], x[1])
            dist_to_boundary = 1.0 - r
            if dist_to_boundary <= eps:
                # project to boundary and evaluate h
                theta = np.arctan2(x[1], x[0])
                estimates[p] = h_boundary(theta, a0, omega_cos, omega_sin, q)
                break
            # sample uniformly on circle of radius dist_to_boundary about x
            phi = 2.0*np.pi*np.random.rand()
            x = x + dist_to_boundary * np.array([np.cos(phi), np.sin(phi)])
        else:
            # max steps reached: project to boundary
            theta = np.arctan2(x[1], x[0])
            estimates[p] = h_boundary(theta, a0, omega_cos, omega_sin, q)
    return estimates.mean(), estimates.std()/np.sqrt(n_paths)

def wos_vectorized(points, n_paths, eps=1e-3, max_steps=10000, batch=256):
    """
    Compute WOS estimate for many points.
    points: (M,2) array
    Returns means (M,) and sems (M,) (standard error of mean).
    This vectorized approach loops over points but vectorizes paths per point for simplicity.
    """
    M = points.shape[0]
    means = np.zeros(M)
    sems = np.zeros(M)
    for i in range(M):
        means[i], sems[i] = wos_single_point(points[i], n_paths, eps=eps, max_steps=max_steps)
    return means, sems

# -----------------------------
# Run WOS on interior points (this can be slow; we demo on a sparse grid)
# -----------------------------
# For demonstration, pick a coarse set of evaluation points to keep runtime reasonable.
# You can increase resolution or number of paths as needed.
eval_mask = inside_mask.copy()
# optionally subsample interior points for speed (comment out to use all)
# take every kth interior point
k = 6
idx_in = np.where(inside_mask)[0][::k]
pts_eval = points[idx_in]

print("Evaluating WOS at {} interior points with {} paths per point...".format(len(pts_eval), n_paths))

means, sems = wos_vectorized(pts_eval, n_paths=n_paths, eps=eps, max_steps=max_steps)

# analytic values at same points
analytic_vals = analytic_solution(pts_eval, a0, omega_cos, omega_sin, q)

# -----------------------------
# Prepare plotting arrays (grid)
# -----------------------------
# We'll build full grids for plotting analytic and approximate fields inside the disk.
U_analytic = np.full(points.shape[0], np.nan)
U_wos = np.full(points.shape[0], np.nan)
U_se = np.full(points.shape[0], np.nan)

U_analytic[idx_in] = analytic_vals
U_wos[idx_in] = means
U_se[idx_in] = sems

# reshape to grid for plotting
U_analytic_grid = U_analytic.reshape(nx, ny)
U_wos_grid = U_wos.reshape(nx, ny)
U_se_grid = U_se.reshape(nx, ny)

# -----------------------------
# Visualization
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12,10))
ax = axes[0,0]
im = ax.imshow(U_analytic_grid.T, origin='lower', extent=[-1,1,-1,1], cmap='viridis')
ax.set_title("Analytic harmonic solution")
plt.colorbar(im, ax=ax)

ax = axes[0,1]
im = ax.imshow(U_wos_grid.T, origin='lower', extent=[-1,1,-1,1], cmap='viridis')
ax.set_title("WOS Monte Carlo estimate (subsampled points)")
plt.colorbar(im, ax=ax)

ax = axes[1,0]
# absolute error at evaluated points (set nan to zero for plotting)
err_grid = np.abs(U_wos_grid - U_analytic_grid)
im = ax.imshow(err_grid.T, origin='lower', extent=[-1,1,-1,1], cmap='inferno')
ax.set_title("Absolute error (WOS vs analytic)")
plt.colorbar(im, ax=ax)

ax = axes[1,1]
# plot a few WOS sample paths starting from one location for illustration
start = np.array([0.2, -0.3])
n_paths_demo = 6
ax.plot(start[0], start[1], 'ko', label='start')
for p in range(n_paths_demo):
    # simulate path and record
    x = start.copy()
    pathx = [x[0]]; pathy = [x[1]]
    for step in range(1000):
        r = np.hypot(x[0], x[1])
        d = 1.0 - r
        if d <= eps:
            theta = np.arctan2(x[1], x[0])
            bx = np.cos(theta); by = np.sin(theta)
            pathx.append(bx); pathy.append(by)
            break
        phi = 2*np.pi*np.random.rand()
        x = x + d * np.array([np.cos(phi), np.sin(phi)])
        pathx.append(x[0]); pathy.append(x[1])
    ax.plot(pathx, pathy, '-o', markersize=3, alpha=0.7)
# draw unit circle
th = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(th), np.sin(th), 'k-', lw=1)
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05,1.05)
ax.set_title("Some WOS sample paths (demo)")

plt.tight_layout()
plt.show()

# -----------------------------
# Print some diagnostics
# -----------------------------
abs_err = np.abs(means - analytic_vals)
print("WOS per-point RMSE over evaluated points: {:.3e}".format(np.sqrt(np.mean(abs_err**2))))
print("Mean reported SEM (std error) per point: {:.3e}".format(np.nanmean(sems)))
