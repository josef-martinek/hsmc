import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm

# === Problem setup ===
# Large computational domain to approximate R
x_min, x_max = -2.0, 2.0
nx = 1000  # finer grid since domain is larger
dx = (x_max - x_min) / nx
mesh = Grid1D(dx=dx, nx=nx)

# Shift mesh cell centers so x ∈ [x_min, x_max]
x = mesh.cellCenters[0].value + x_min

# Spatially varying diffusivity
D_field = np.exp(0.1 + 0.5 * np.cos(2.0 * np.pi * x ))#/ (x_max - x_min))
D = CellVariable(mesh=mesh, value=D_field)

# Initial condition: Gaussian bump at x0
x0 = 0.0
sigma0 = 0.05
p0 = np.exp(-((x - x0) ** 2) / (2 * sigma0**2))
#p0 /= np.trapz(p0, x)  # normalize over [-5, 5]
p = CellVariable(mesh=mesh, value=p0, name="p")

# PDE: ∂t p = ∂x²(D p)
eq = TransientTerm() == DiffusionTerm(coeff=D)

# Time stepping
dt = 1e-4
t_final = 0.05
n_steps = int(t_final / dt)

# Store snapshots
snapshots = [p.value.copy()]
times = [0.0]
save_times = [0.001, 0.005, 0.01, 0.02, 0.05]

for step in range(n_steps):
    eq.solve(var=p, dt=dt)
    t = (step + 1) * dt
    if np.isclose(t, save_times, atol=0.001 * dt).any():
        snapshots.append(p.value.copy())
        times.append(t)

# === Check and print boundary values ===
left_val = p.value[0]
right_val = p.value[-1]
print(f"Boundary values at t={t_final:.3f}: p(x_min) = {left_val:.3e}, p(x_max) = {right_val:.3e}")

# === Truncate and plot results on [-1, 1] ===
x_plot_min, x_plot_max = -1.0, 1.0
mask = (x >= x_plot_min) & (x <= x_plot_max)
x_plot = x[mask]

plt.figure(figsize=(7, 4))
for pi, ti in zip(snapshots, times):
    plt.plot(x_plot, pi[mask], label=f"t = {ti:.3f}")
plt.title("Solution of ∂t p = ∂x²(D p) using FiPy (truncated to [-1,1])")
plt.xlabel("x")
plt.ylabel("p(x,t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
