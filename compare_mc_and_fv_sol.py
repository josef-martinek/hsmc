import numpy as np
import problem_setup.fokker_planck.likelihood_model as lm
import matplotlib.pyplot as plt

np.random.seed(seed=1)

n_params = 1
s = 5
params = np.zeros((n_params, s))
results_em = lm.FPModelHD.euler_maruyama_binned_vectorized(params=params,
                                                           nsamp=1000)
results_em = results_em.flatten()
results_fv = lm.FPModelHD.fv_solution(params=params.flatten())

x_plot = np.linspace(-1, 1, lm.FPModelHD.n_bins + 2)[1:-1]

plt.figure(figsize=(7, 4))
plt.plot(x_plot, results_em, label=f"Monte Carlo")
plt.plot(x_plot, results_fv, label=f"Finite volumes")
plt.title("Solution of ∂t p = ∂x²(D p) (truncated to [-1,1])")
plt.xlabel("x")
plt.ylabel("p(x,t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()