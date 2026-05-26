from particles.smc_samplers import ArrayMetropolis, ArrayRandomWalk, view_2d_array
from particles.smc_samplers import AdaptiveMCMCSequence, MCMCSequence
from particles import resampling as rs
import numpy as np
from scipy import stats, linalg
from numpy import random


class RWRegCal(ArrayRandomWalk):

    def __init__(self, eye_prop=False):
        super().__init__()
        self.eye_prop = eye_prop

    def calibrate(self, W, x):
        arr = view_2d_array(x.theta)
        N, d = arr.shape
        m, cov = rs.wmean_and_cov(W, arr)
        cov_reg = cov + 1e-8 * np.eye(N=cov.shape[0])
        scale = 2.38 / np.sqrt(d)
        if self.eye_prop:
            x.shared["chol_cov"] = scale * np.eye(N=cov.shape[0])
        else:
            x.shared["chol_cov"] = scale * linalg.cholesky(cov_reg, lower=True)


class RWUnc(ArrayMetropolis):
    # The method calibrate() from ArrayMetropolis is called, but the results are not used
    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def proposal(self, x, xprop):
        arr = view_2d_array(x.theta)
        N, d = arr.shape
        scale = 2.38 / np.sqrt(d)
        scale = scale*self.scale_factor  # Set up MH stepsize
        arr_prop = view_2d_array(xprop.theta)
        arr_prop[:, :] = arr + scale*stats.norm.rvs(size=arr.shape)
        return 0.0


class NoisyMetropolisUnc(RWUnc):

    def step(self, x, target=None):
        xprop = x.__class__(theta=np.empty_like(x.theta))
        delta_lp = self.proposal(x, xprop)
        target(xprop)
        target(x)  # Here is where the lik. reeval. is set up
        lp_acc = xprop.lpost - x.lpost + delta_lp
        pb_acc = np.exp(np.clip(lp_acc, None, 0.0))
        mean_acc = np.mean(pb_acc)
        accept = random.rand(x.N) < pb_acc
        x.copyto(xprop, where=accept)
        return mean_acc
    

class NoisyMetropolisCal(ArrayRandomWalk):

    def __init__(self, scale=1.0):
        self.scale = scale

    def step(self, x, target=None):
        xprop = x.__class__(theta=np.empty_like(x.theta))
        delta_lp = self.proposal(x, xprop)
        target(xprop)
        target(x)  # Here is where the lik. reeval. is set up
        lp_acc = xprop.lpost - x.lpost + delta_lp
        pb_acc = np.exp(np.clip(lp_acc, None, 0.0))
        mean_acc = np.mean(pb_acc)
        accept = random.rand(x.N) < pb_acc
        x.copyto(xprop, where=accept)
        return mean_acc

    def proposal(self, x, xprop):
        L = x.shared["chol_cov"]
        arr = view_2d_array(x.theta)
        arr_prop = view_2d_array(xprop.theta)
        arr_prop[:, :] = arr + self.scale * stats.norm.rvs(size=arr.shape) @ L.T
        return 0.0



class TwoStageMCMC(MCMCSequence):

    def __init__(self,
                 mcmc1=None,
                 len_chain1=10,
                 mcmc2=None,
                 len_chain2=10,
                 coarse_model=None,
                 fine_over_coarse=10):
        self.mcmc1 = ArrayRandomWalk() if mcmc1 is None else mcmc1
        self.len_chain1 = len_chain1
        self.nsteps1 = len_chain1 - 1
        self.mcmc2 = ArrayRandomWalk() if mcmc2 is None else mcmc2
        self.len_chain2 = len_chain2
        self.nsteps2 = len_chain2 - 1
        self.coarse_model = coarse_model
        self.fine_over_coarse = fine_over_coarse
        self.coarse_target = None

    def calibrate(self, W, x):
        nsamp_fine = x.shared["solver_nsamp"][-1]
        epn = x.shared["exponents"][-1]
        self.coarse_model.nsamp = nsamp_fine / self.fine_over_coarse

        def func(x_aux):
            x_aux.lprior = self.coarse_model.prior.logpdf(x_aux.theta)
            x_aux.llik = self.coarse_model.loglik(x_aux.theta)
            if epn > 0.0:
                x_aux.lpost = x_aux.lprior + epn * x_aux.llik
            else:  # avoid having 0 x Nan
                x_aux.lpost = x_aux.lprior.copy()

        self.coarse_target = func

        self.mcmc2.calibrate(W, x)

    def __call__(self, x, target_fine):
        xout = x.copy()
        ars = []
        for _ in range(self.nsteps1):
            ar = self.mcmc1.step(xout, self.coarse_target)
            #print("MCWM ar:", ar)
        if self.nsteps1 > 0:
            target_fine(xout)
        for _ in range(self.nsteps2):
            ar = self.mcmc2.step(xout, target_fine)
            #print("PMMH ar:", ar)
            ars.append(ar)
        prev_ars = x.shared.get("acc_rates", [])
        xout.shared["acc_rates"] = prev_ars + [ars]  # a list of lists
        return xout
