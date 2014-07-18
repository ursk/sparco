"""
different learning methods

1. Jack's angle chasing method
2. Coordinate-wise descent
3. Parallel block coordinate-wise descent
4. Truncated projected gradient
"""
import numpy as np
from time import time as now
import mpi4py.rc
mpi4py.rc.profile('MPE')
from mpi4py import MPI, MPE
import scipy.sparse as sparse
from scipy.signal import lfilter
# import ipdb
import h5py
import os

import mpi

class Learner(object):
  pass

  def angles(self, phi0, phi):
      """
      Returns angles between basis vectors phi0 and phi in
      degrees. Assumes norm one.
      """
      dots = np.array([np.sum(phi[:,i]*phi0[:,i])
                       for i in range(phi.shape[1])])
      angle = np.arccos(dots) * 180. / np.pi
      angle[np.isnan(angle)] = 0.
      return angle
