import numpy as np
import matplotlib.pyplot as plt
from problem_setup.wos.likelihood_model import WOSHD  # <-- adjust path if needed
import time

# one parameter vector (s cos coeffs)
#params = np.array([[1.0, -0.5, 0.3, 0.0, 0.0, 0.0]])
np.random.seed(seed=4)
params = np.random.randn(1, 20)
npaths = 2

start = time.time()
print("Number of inner points:", WOSHD.points_in().size)
u_mc = WOSHD.wos(params, npaths=npaths)    # (n_points, 1)
time_mc = time.time()
print("WOS simulation time:", time_mc-start)
u_ex = WOSHD.exact_sol(params)
time_ex = time.time()
print("Exact solution time:", time_ex-time_mc)

# visualize on grid
grid_all = WOSHD.grid_all()
mask = WOSHD.mask_inside()
nx, ny = WOSHD.nx, WOSHD.ny

U_mc = np.full(grid_all.shape[0], np.nan)
U_ex = np.full_like(U_mc, np.nan) 
U_mc[mask] = u_mc[:, 0]
U_ex[mask] = u_ex[:, 0]

U_mc_grid = U_mc.reshape(nx, ny)
U_ex_grid = U_ex.reshape(nx, ny)

plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(U_ex_grid.T, origin='lower', extent=[-1,1,-1,1])
plt.title('Exact')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(U_mc_grid.T, origin='lower', extent=[-1,1,-1,1])
plt.title('WOS MC')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(np.abs(U_ex_grid - U_mc_grid).T, origin='lower', extent=[-1,1,-1,1])
plt.title('Abs error')
plt.colorbar()
plt.tight_layout()
plt.show()

# numeric error on interior points
err = u_mc[:,0] - u_ex[:,0]
print("RMSE:", np.sqrt(np.mean(err**2)))