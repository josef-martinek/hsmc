
from juliacall import Main as jl
import matplotlib.pyplot as plt
import time
import numpy as np

jl.seval("include(\"problem_setup/rayleigh_problem_dsmc/rayleigh.jl\")")

start_time = time.time()
ps, vs, params = jl.simulate(Kn=0.1, Nsim=10, Nmft=10, Ncell=32)
print(f"Simulation time in seconds: {time.time()-start_time}")

mac = jl.stats(ps, vs, params)

# Extract arrays
z = np.array(mac.z)
uy = np.array(mac.uy)

# Save both into a single file
np.savez("dsmc_data.npz", z=z, uy=uy)

# fig, ax1 = plt.subplots()
# ax1.plot(mac.z, mac.uy, label='u_y')
# ax1.set_xlabel('z')
# ax1.set_ylabel('Velocity')
# ax1.legend()
# plt.title('Rayleigh problem')
# plt.show()

