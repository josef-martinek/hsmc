from particles.smc_samplers import AdaptiveTempering, next_annealing_epn
import numpy as np
from particles.smc_samplers import ThetaParticles

class TempAdaptNsampRew(AdaptiveTempering):

    def __init__(self, Nmin, ar_tol, model=None, wastefree=False, len_chain=10, move=None, ESSrmin=0.5, max_iter=1000):
        super().__init__(model, wastefree, len_chain, move, ESSrmin, max_iter)
        self.Nmin = Nmin
        self.ar_tol = ar_tol
        self.solver_nsamp = [self.Nmin]

    def _M0(self, N):
        x0 = ThetaParticles(theta=self.model.prior.rvs(size=N))
        x0.shared["exponents"] = [0.0]
        x0.shared["path_sampling"] = [0.0]
        x0.shared["solver_nsamp"] = [self.solver_nsamp[0]]
        self.current_target(0.0)(x0)
        return x0

    def logG(self, t, xp, x):
        epn = x.shared['exponents'][-1]
        new_epn = next_annealing_epn(epn, self.ESSrmin, x.llik)
        x.shared['exponents'].append(new_epn)

        try:
            mean_ar = np.mean(x.shared["acc_rates"][-1])
        except (KeyError, IndexError):
            mean_ar = 1.0
        if mean_ar < self.ar_tol:
            last_nsamp = self.solver_nsamp[-1]
            self.solver_nsamp.append(last_nsamp*2)
            x.shared["solver_nsamp"].append(last_nsamp*2)
        else:
            last_nsamp = self.solver_nsamp[-1]
            self.solver_nsamp.append(last_nsamp)
            x.shared["solver_nsamp"].append(last_nsamp)

        return self.logG_tempering(x, new_epn - epn)
    
    def logG_tempering(self, x, delta):
        dl = delta * x.llik
        x.lpost += dl
        #self.update_path_sampling_est(x, delta)

        dl2 = 0
        if self.solver_nsamp[-1] != self.solver_nsamp[-2]:
            llik_new = self.model.loglik(x.theta, self.solver_nsamp[-1])
            dl2 = x.shared['exponents'][-1] * (llik_new - x.llik)
            x.lpost += dl2
            x.llik = llik_new
        return dl + dl2
    
    def current_target(self, epn):
        def func(x):
            x.lprior = self.model.prior.logpdf(x.theta)
            x.llik = self.model.loglik(x.theta, self.solver_nsamp[-1])
            if epn > 0.0:
                x.lpost = x.lprior + epn * x.llik
            else:  # avoid having 0 x Nan
                x.lpost = x.lprior.copy()

        return func
    

class TempAdaptNsampNoRew(TempAdaptNsampRew):

    def logG_tempering(self, x, delta):
        dl = delta * x.llik
        x.lpost += dl
        #self.update_path_sampling_est(x, delta)

        dl2 = 0
        if self.solver_nsamp[-1] != self.solver_nsamp[-2]:
            llik_new = self.model.loglik(x.theta, self.solver_nsamp[-1])
            dl2 = x.shared['exponents'][-1] * (llik_new - x.llik)
            x.lpost += dl2
            x.llik = llik_new
        return dl
    

class TempAdaptNsampNoIS(TempAdaptNsampRew):

    def logG_tempering(self, x, delta):
        dl = delta * x.llik
        x.lpost += dl
        #self.update_path_sampling_est(x, delta)
        return dl