from particles.smc_samplers import ArrayMetropolis, ArrayRandomWalk, ArrayMCMC, view_2d_array
from particles.smc_samplers import AdaptiveMCMCSequence
from particles import resampling as rs
import numpy as np
from scipy import stats, linalg
from numpy import random


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


class DA(ArrayMCMC):

    def __init__(self,
                 coarse_chain: AdaptiveMCMCSequence,
                 coarse_model= None, #Should be ModelInexactAdapt
                 fine_over_coarse=10):
        self.coarse_chain = coarse_chain
        self.coarse_model = coarse_model
        self.fine_over_coarse = fine_over_coarse
        self.coarse_target = None

    def calibrate(self, W, x):
        arr = view_2d_array(x.theta)
        N, d = arr.shape
        m, cov = rs.wmean_and_cov(W, arr)
        scale = 2.38 / np.sqrt(d)
        x.shared["chol_cov"] = scale * linalg.cholesky(cov, lower=True)

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

    def step(self, x, target_fine=None):
        xprop = self.proposal(x, self.coarse_target)
        if np.allclose(x.theta['param'], xprop.theta['param']):
            pb_acc = np.ones_like(xprop.lpost)
        else:
            target_fine(xprop)
            lp_acc = xprop.lpost - x.lpost
            pb_acc = np.exp(np.clip(lp_acc, None, 0.0))
        mean_acc = np.mean(pb_acc)
        accept = random.rand(x.N) < pb_acc
        x.copyto(xprop, where=accept)
        return mean_acc

    def proposal(self, x, target=None):
        xin = x.copy()
        xprop = self.coarse_chain(xin, target=target)
        return xprop
