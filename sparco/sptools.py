"""
some random tools, slow code.
"""
import os

import numpy as np
import scipy.signal as signal

DEBUG = True
def debug(str):
    if DEBUG: print "(%s)" % str

def vnorm(phi):
    """
    Norm of each basis element as a [1, N, 1] matrix
    """
    return np.sqrt((phi**2).sum(axis=0).sum(axis=1))[np.newaxis,:,np.newaxis]

def norm(phi):
    """
    Norm of each basis element as a N-vector
    """
    return np.sqrt((phi**2).sum(axis=0).sum(axis=1))

def attributesFromDict(d):
    "Automatically initialize instance variables, Python Cookbook 6.18"
    self = d.pop('self')
    for n, v in d.iteritems():
        setattr(self, n, v)

def weighted_randint(weights, size=1):
    """
    Returns size random integers from 0 to len(weights)-1
    weighted with weights
    """
    s = np.nonzero(np.random.multinomial(1, weights, size=size))[1]
    if size == 1:
        return s[0]
    else:
        return s

def merge(*dicts):
  """Merge an arbitrary number of dictionaries.

  Values in dictionaries occurring later in the argument list have priority.

  Args:
    *dicts: Arbitrary number of configuration dictionaries.
  """
  def mergeInner(config1, config2):
    for k,v in config2.items():
      if type(v) is dict and config1.has_key(k):
        config1[k].update(config2[k])
      else:
        config1[k] = config2[k]
    return config1
  return reduce(mergeInner, dicts)

def blur(phi, window=.2):
    """
    Gaussian blur of basis functions
    """
    C, N, P = phi.shape
    w = np.int(window * min(C,P))
    g = signal.gaussian(w, 1)
    philp = np.empty_like(phi)
    for i in range(N):
        philp[:,i] = signal.sepfir2d(phi[:,i], g, g)
    return philp

