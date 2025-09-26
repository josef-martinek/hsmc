import numpy as np
import matplotlib.pyplot as plt

# Load arrays
#data = np.load("problem_setup/rayleigh_problem_dsmc/ground_truth/dsmc_data.npz")
data = np.load("dsmc_data.npz")
z = data["z"]
uy = data["uy"]

# Plot
print(uy)
fig, ax1 = plt.subplots()
ax1.plot(z, uy, label="u_y")
ax1.set_xlabel("z")
ax1.set_ylabel("Velocity")
ax1.legend()
plt.title("Rayleigh problem")
plt.show()