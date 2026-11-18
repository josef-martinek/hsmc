from hsmc.samplers import TempNsampNoRew

class DensityTempNsampNoRew(TempNsampNoRew):
    """
    Custom SMC Sampler for the density problem. It inherits from TempNsampNoRew
    and adds exact density tracking. The noise realization `e_arr` is saved
    as a dedicated particle attribute outside of the `theta` structured array
    to avoid breaking the standard MCMC mechanics.
    """
    def current_target(self, epn):
        def func(x):
            x.lprior = self.model.prior.logpdf(x.theta)
            x.llik = self.model.loglik(x.theta, self.solver_nsamp_last[-1])
            
            # Fetch the generated 'e' from the model and attach it to the particles
            x.e_arr = self.model.last_e_eval
            
            if epn > 0.0:
                x.lpost = x.lprior + epn * x.llik
            else:
                x.lpost = x.lprior.copy()
        return func

    def logG_tempering(self, x, delta):
        # 1. Standard tempering weight based on current fixed potential
        dl = delta * x.llik
        x.lpost += dl

        # 2. Add custom kernel ratio penalty if fidelity changed
        shared_nsamp = x.shared.get("solver_nsamp", [])
        if len(shared_nsamp) >= 2 and shared_nsamp[-1] != shared_nsamp[-2]:
            S_prev = shared_nsamp[-2]
            S_curr = shared_nsamp[-1]
            e_particles = x.e_arr
            log_ratio = self.model.evaluator.evaluate_log_ratio(e_particles, S_prev, S_curr)
            
            # Apply exact density penalty
            dl += log_ratio
            
        return dl
