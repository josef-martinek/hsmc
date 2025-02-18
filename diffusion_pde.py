import numpy as np
import matplotlib.pyplot as plt
import sdepy

# Parameters
D = 1.0  # Diffusion coefficient
t = 1  # Time to evaluate
timestep = 0.01
x_min, x_max = -5, 5  # Spatial range
n_bins = 1000  # Number of bins for histogram
n_particles = 100000  # Number of particles
num_int = round(t/timestep)

# Initial condition: Dirac delta at x=0
initial_position = 0.0

# Define the SDE: dX_t = sqrt(2D) dW_t
def drift(t, x):
    return 0  # No drift

def diffusion(t, x):
    return np.sqrt(2 * D)  # Constant diffusion

def euler_maruyama(f, g, x0, t, dt):
    n = len(t)
    x = np.zeros((len(X0),n))
    Dt = np.zeros(len(X0))
    x[:,0] = x0
    for i in range(1, n):
        dw = np.sqrt(dt) * np.random.normal(size=(len(X0)))
        #x[:,i] = x[:,i - 1] + f(x[:,i - 1], t[i - 1]) * dt + g(x[:,i - 1], t[i - 1]) * dw
        x[:,i] = x[:,i - 1] + g(x[:,i - 1], t[i - 1]) * dw
    return x

# Simulate particle trajectories
X0 = np.full(n_particles, initial_position)  # All particles start at x=0
times = np.linspace(0, t, num_int)  # Start and end times
trajectories = euler_maruyama(f=drift, g=diffusion, x0=X0, t=times, dt=timestep)

# Extract final positions
final_positions = trajectories[:,-1]

# Histogram to approximate solution
bins = np.linspace(x_min, x_max, n_bins)
hist, bin_edges = np.histogram(final_positions, bins=bins, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Analytical solution
x_vals = np.linspace(x_min, x_max, n_bins)
u_analytical = (1 / np.sqrt(4 * np.pi * D * t)) * np.exp(-x_vals**2 / (4 * D * t))

# Plot comparison
plt.figure(figsize=(8, 5))
plt.plot(bin_centers, hist, label='SDE Solution (Histogram)', lw=2)
plt.plot(x_vals, u_analytical, label='Analytical Solution', lw=2, linestyle='--')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Comparison of SDE and Analytical Solution')
plt.legend()
plt.grid()
plt.show()
