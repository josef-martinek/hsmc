from particles.smc_samplers import AdaptiveTempering, next_annealing_epn, Tempering
import numpy as np
from particles.smc_samplers import ThetaParticles

class TempAdaptNsampNoRew(AdaptiveTempering):
    """
    Adaptive Tempering SMC sampler with adaptive solver sample size (no reweighting).
    """

    def __init__(self, Nmin, ar_tol, model=None, wastefree=False, len_chain=10, move=None, ESSrmin=0.5, max_iter=1000, growth_factor=2, Nmax=None):
        if Nmax is not None and Nmax < Nmin:
            raise ValueError("Nmax must be greater than or equal to Nmin")
        super().__init__(model, wastefree, len_chain, move, ESSrmin, max_iter)
        self.Nmin = Nmin
        self.Nmax = Nmax
        self.ar_tol = ar_tol
        self.solver_nsamp = [self.Nmin]
        self.growth_factor = growth_factor

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
        last_nsamp = self.solver_nsamp[-1]
        proposed_nsamp = last_nsamp * self.growth_factor
        should_increase = mean_ar < self.ar_tol and (
            self.Nmax is None or proposed_nsamp <= self.Nmax
        )
        next_nsamp = proposed_nsamp if should_increase else last_nsamp
        self.solver_nsamp.append(next_nsamp)
        x.shared["solver_nsamp"].append(next_nsamp)

        return self.logG_tempering(x, new_epn - epn)
    
    def logG_tempering(self, x, delta):
        dl = delta * x.llik
        x.lpost += dl

        if self.solver_nsamp[-1] != self.solver_nsamp[-2]:
            llik_new = self.model.loglik(x.theta, self.solver_nsamp[-1])
            dl2 = x.shared['exponents'][-1] * (llik_new - x.llik)
            x.lpost += dl2
            x.llik = llik_new
        return dl
    
    def current_target(self, epn):
        def func(x):
            x.lprior = self.model.prior.logpdf(x.theta)
            x.llik = self.model.loglik(x.theta, self.solver_nsamp[-1])
            if epn > 0.0:
                x.lpost = x.lprior + epn * x.llik
            else:  # avoid having 0 x Nan
                x.lpost = x.lprior.copy()

        return func

class TempNsampNoRew(Tempering):
    """
    Fixed Tempering SMC sampler with a fixed schedule for solver sample size.
    Does not use importance reweighting for likelihood changes.
    """
    def __init__(self, model=None, wastefree=False, len_chain=10, move=None, exponents=None, Nsamps=None):
        super().__init__(model, wastefree, len_chain, move, exponents)
        self.solver_nsamp = Nsamps
        self.solver_nsamp_last = [Nsamps[0]]

    def _M0(self, N):
        x0 = ThetaParticles(theta=self.model.prior.rvs(size=N))
        x0.shared["exponents"] = [0.0]
        x0.shared["path_sampling"] = [0.0]
        x0.shared["solver_nsamp"] = [self.solver_nsamp[0]]
        self.current_target(0.0)(x0)
        return x0

    def logG(self, t, xp, x):
        x.shared['exponents'].append(self.exponents[t])
        x.shared["solver_nsamp"].append(self.solver_nsamp[t+1])
        self.solver_nsamp_last.append(self.solver_nsamp[t+1])

        return self.logG_tempering(x, self.deltas[t])
    
    def logG_tempering(self, x, delta):
        dl = delta * x.llik
        x.lpost += dl

        dl2 = 0
        if  x.shared["solver_nsamp"][-1] !=  x.shared["solver_nsamp"][-2]:
            llik_new = self.model.loglik(x.theta, x.shared["solver_nsamp"][-1])
            dl2 = x.shared['exponents'][-1] * (llik_new - x.llik)
            x.lpost += dl2
            x.llik = llik_new
        return dl
    
    def current_target(self, epn):
        def func(x):
            x.lprior = self.model.prior.logpdf(x.theta)
            x.llik = self.model.loglik(x.theta, self.solver_nsamp_last[-1])
            if epn > 0.0:
                x.lpost = x.lprior + epn * x.llik
            else:  # avoid having 0 x Nan
                x.lpost = x.lprior.copy()

        return func
