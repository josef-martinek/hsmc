import numpy as np
import matplotlib.pyplot as plt
from problem_setup.wos.likelihood_model import WOSHD
from problem_setup.wos.problem_parameters import SamplerConfig as sc
import time
import os

# ---------------------------------------------------------
# 1. Load truth and run simulations
# ---------------------------------------------------------
#params = sc.ground_truth_par
#params = np.zeros(shape=(32,))

results_folder = "data/run_20251113-103422_1_chosen_experiments"
ref_path = os.path.join(results_folder, "post_mean_ref.npz")
ref_data = np.load(ref_path)
params = ref_data["post_mean"]

npaths = 5

start = time.time()
print("Number of inner points:", WOSHD.points_in().shape[0])
u_mc = WOSHD.wos(params, npaths=npaths)
time_mc = time.time()
print("WOS simulation time:", time_mc - start)

u_ex = WOSHD.exact_sol(params)
time_ex = time.time()
print("Exact solution time:", time_ex - time_mc)

# Simulated noisy data (same as in inference)
rng_data = np.random.default_rng(seed=81)
observed_data = u_ex + rng_data.normal(
    loc=0., scale=np.sqrt(sc.data_noise_var), size=u_ex.shape
)

# ---------------------------------------------------------
# 2. Build full grid values safely
# ---------------------------------------------------------
grid = WOSHD.grid_all()        # (nx*ny, 2)
nx, ny = WOSHD.nx, WOSHD.ny

pts_in = WOSHD.points_in()     # coordinates used in WOS simulation
pts_in_tuple = set(map(tuple, pts_in))

# Build mask_sparse safely by coordinate equality
mask_sparse = np.array([tuple(p) in pts_in_tuple for p in grid])

# Allocate output arrays
U_mc = np.full(grid.shape[0], np.nan)
U_ex = np.full(grid.shape[0], np.nan)
U_obs = np.full(grid.shape[0], np.nan)

# Assign simulated values
U_mc[mask_sparse]  = u_mc[:, 0]
U_ex[mask_sparse]  = u_ex[:, 0]
U_obs[mask_sparse] = observed_data[:, 0]

# Reshape to 2D
U_mc_grid  = U_mc.reshape(nx, ny)
U_ex_grid  = U_ex.reshape(nx, ny)
U_obs_grid = U_obs.reshape(nx, ny)

# ---- Visualization ----
plt.figure(figsize=(13, 4))

plt.subplot(1, 2, 1)
plt.imshow(U_ex_grid.T, origin='lower', extent=[-1,1,-1,1])
plt.title('Exact')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(U_obs_grid.T, origin='lower', extent=[-1,1,-1,1])
plt.title('Observed data')
plt.colorbar()

# plt.subplot(1, 4, 3)
# plt.imshow(U_mc_grid.T, origin='lower', extent=[-1,1,-1,1])
# plt.title('WOS MC')
# plt.colorbar()

# plt.subplot(1, 4, 4)
# plt.imshow(np.abs(U_ex_grid - U_mc_grid).T, origin='lower', extent=[-1,1,-1,1])
# plt.title('Abs error')
# plt.colorbar()

plt.tight_layout()
plt.show()

# ---- Numeric error on interior points ----
err = u_mc[:,0] - u_ex[:,0]
print("RMSE (MC vs Exact):", np.sqrt(np.mean(err**2)))