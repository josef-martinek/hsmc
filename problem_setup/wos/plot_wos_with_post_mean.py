import numpy as np
import matplotlib.pyplot as plt
from problem_setup.wos.likelihood_model import WOSHD
from problem_setup.wos.problem_parameters import SamplerConfig as sc
import time
import os

# ---------------------------------------------------------
# 1. Load truth and run simulations
# ---------------------------------------------------------
params = sc.ground_truth_par
#params = np.zeros(shape=(32,))

#results_folder = "data/run_20251113-103422_1_chosen_experiments"
#ref_path = os.path.join(results_folder, "post_mean_ref.npz")
#ref_data = np.load(ref_path)
#params = ref_data["post_mean"]

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
# 2. Visualisation: exact field + measurement points
# ---------------------------------------------------------
grid = WOSHD.grid_all()        # (nx*ny, 2)
nx, ny = WOSHD.nx, WOSHD.ny

pts_in = WOSHD.points_in()     # (M,2) measurement points (polar uniform)
u_pts  = u_mc[:, 0]            # WOS MC values at measurement points
u_true = u_ex[:, 0]            # exact values at measurement points
u_obsv = observed_data[:, 0]   # noisy observed values at measurement points

# ---- Create exact solution only on disk grid points ----
u_ex_pts = WOSHD.exact_sol_full(params)[:, 0]   # (K,) values
grid = WOSHD.grid_all()
r2 = grid[:, 0]**2 + grid[:, 1]**2
pts_inside = grid[r2<1.0]                         # (K,2) matching u_ex_pts

# ---- Plot ----
plt.figure(figsize=(5, 4))
sc = plt.scatter(pts_inside[:,0], pts_inside[:,1],
                 c=u_ex_pts, s=20, cmap='viridis')
plt.colorbar(sc, label='Exact solution')
plt.title('Exact solution')

# Overlay measurement points
plt.scatter(pts_in[:, 0], pts_in[:, 1],
            marker='x', c='black', s=22, linewidths=1.0,
            label='measurement points')

plt.legend(loc='upper right')
plt.axis('equal')
plt.tight_layout()
plt.show()

