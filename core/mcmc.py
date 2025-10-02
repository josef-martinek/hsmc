from particles.smc_samplers import ArrayMetropolis, ArrayRandomWalk, view_2d_array
from particles import resampling as rs
import numpy as np
from scipy import linalg, stats
from numpy import random


class RWUncalibrated(ArrayMetropolis):

    def proposal(self, x, xprop):
        arr = view_2d_array(x.theta)
        N, d = arr.shape
        scale = 2.38 / np.sqrt(d)
        scale = scale/4 #Set up MH stepsize
        arr_prop = view_2d_array(xprop.theta)
        arr_prop[:, :] = arr + scale*stats.norm.rvs(size=arr.shape)
        return 0.0


class NoisyMetropolis(RWUncalibrated):

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