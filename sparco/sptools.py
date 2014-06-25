"""
some random tools, slow code.
"""
from IPython import embed

import os
import time
import types

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import sparco.mpi as mpi

DEBUG = True

class data:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

def mixin(instance, mixin_class):
  """Dynamic mixin of methods into instance - works only for new style classes"""
  for name in mixin_class.__dict__:
    if name.startswith('__') and name.endswith('__'):
      continue
    elif not type(mixin_class.__dict__[name])==types.FunctionType:
      continue
    else:
      instance.__dict__[name]=mixin_class.__dict__[name].__get__(instance)

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

def sample_array(ary, length, axis=0):
  max_start = ary.shape[axis] - length + 1
  start = np.random.randint(0, max_start)
  slc = (slice(None),)*ary.ndims
  slc[axis] = slice(start, start+length)
  return ary[slc]

def generate_filtered(gen, filt, num):
  res = []
  while len(res) < num:
    x = gen()
    filt(x) and res.append(x)
  return res


# computing objective function
# TODO give more generic names, move

def obj(x, a, phi):
  xhat = compute_xhat(phi, a)
  dx = compute_dx(x, xhat=xhat)
  E = compute_E(dx)
  dphi = compute_dphi(dx, a)
  return xhat, dx, E, dphi

def compute_dx(x, a=None, phi=None, xhat=None):
  return x - (xhat if xhat != None else compute_xhat(phi, a))

def compute_xhat(phi, a):
  p = phi.shape[2]; t = a.shape[1] - p + 1
  return reduce(np.add, (np.dot(phi[:,:,i], a[:,i:i+t]) for i in range(p)))

def compute_E(dx):
  return 0.5 * np.linalg.norm(dx)**2

def compute_dphi(dx, a):
  t = dx.shape[1]; p = a.shape[1] - t + 1
  return np.dstack(tuple(np.dot(dx, a[:,i:i+t].T) for i in range(p)))

def compute_angle(phi1, phi2):
  dot = np.sum(phi1*phi2) / (np.linalg.norm(phi1) * np.linalg.norm(phi2))
  angle = np.arccos(dot) * 180 / np.pi
  return 0 if np.isnan(angle) else angle

def compute_proposed_phi(phi, dphi, eta):
  newphi = phi - eta * dphi
  return newphi / vnorm(newphi)

# center and smooth basis
# TODO understand and clean up

def center(arr, maxshift=None):
  """
  Shift each basis function to its center of mass by a maximum
  amount of cmax.

  Optionally, smooth basis functions.

  Center of mass is defined using sum of squares.
  """
  for n in range(self.N):
    s = np.sum(arr[:,n]**2, axis=0)
    total = np.sum(s)
    if total == 0.: continue
    m = int(np.round(np.sum(np.arange(self.P) * s)/total))
    shift = self.P/2 - m
    if maxshift:
      shift = np.sign(shift) * min(abs(shift), self.cmax)
    arr[:,n] = np.roll(arr[:,n], shift, axis=1)
    if shift > 0:
      arr[:,n,0:shift] = 0.
    elif shift < 0:
      arr[:,n,shift:] = 0.
    else:
      continue
    arr[:,n] /= np.linalg.norm(arr[:,n])
    print 'Shifting %d by %d' % (n, shift)

def smooth(phi):
  a = 1
  b = [0.25, .5, 0.25]
  for n in range(self.N):
    phi[:,n] = scipy.signal.lfilter(b, a, phi[:,n], axis=1)


# profiling

PROFILING_TABLE = {}
def time_track(orig):
  PROFILING_TABLE[orig.__name__] = []
  def tracked_function(*args, **kwargs):
    start = time.time()
    res = orig(*args, **kwargs)
    end = time.time()
    PROFILING_TABLE[orig.__name__].append(end - start)
    return res
  return tracked_function if mpi.rank == mpi.root else orig

# TODO complete this terminating decorator
# def terminate_after(seconds):
#   def inner_decorator(orig):
#       @functools.wraps(orig)
#       def wrapper(*args, **kwargs):
#         return test_func(*args, **kwargs)
#       return wrapper
#     return actualDecorator

# plotting
# TODO clean this up and just use matplotlib's subplot functionality

def grid_image(mat, nrows=None, ncols=None, grid_line_width=3, params=None):
  nrows, ncols = compute_grid_dimensions(mat.shape[1], nrows, ncols)
  cell_height, cell_width = mat.shape[0], mat.shape[2]
  img_height = cell_height * ncols + grid_line_width * (ncols+1)
  img_width = cell_width * nrows + grid_line_width * (nrows+1)
  img = np.zeros((img_height, img_width))

  phi, buf, I = mat, grid_line_width, img
  m,n,o = phi.shape
  l0norm, l1norm, variance, l2norm = params
  for i in range(phi.shape[0]):
    # import ipdb
    # flip along channel axis (deep sites at bottom)
    patch = phi[i,::-1]
    sx = (o + buf) * (i % ncols) + buf - 1
    sy = (n + buf) * (i / ncols) + buf - 1
    prange = np.abs(patch).max()
    # rescale patch to [0, 1]
    if prange > .00001: patch = .5 + .5*patch/prange
    else: patch += .5
    I[sy:sy+n, sx:sx+o] = patch
    # add borders
    I[sy-1, sx:sx + max(1,np.round(l1norm[i]*o))] = .6   # top
    I[sy:sy + max(1,np.round(l0norm[i]*n)), sx-1] = .6   # left
    I[sy+n, sx:sx + max(1,np.round(variance[i]*o))] = .6   # bottom
    if l2norm is not None:
      I[sy:sy+max(1,np.round(l2norm[i]*n)), sx+o] = .6      
  return I

def grid_plot(mat, nrows=None, ncols=None, grid_line_width=3, params=None,
              cmap=plt.cm.jet, figno=1, filename=None, title=None):
  image = grid_image(mat, nrows, ncols, grid_line_width, params)
  fig = plt.figure(figno)
  plt.clf()
  plt.imshow(image, cmap=cmap, interpolation='nearest', aspect='equal',
             origin='upper',vmax=1, vmin=0)
  if title:
    plt.title(title)
  plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.95,
            wspace=0.02, hspace=0.02)
  plt.draw()
  # save figure  
  if filename:
    dpi = max(150, np.int(2*I.shape[0]/fig.get_figheight()))
    plt.savefig(filename, dpi=dpi)


def compute_grid_dimensions(min_cells, nrows=None, ncols=None):
  if nrows == None and ncols == None:
    root = np.sqrt(min_cells)
    return (np.ceil(root), root)
  elif nrows == None:
    return np.ceil(min_cells / ncols), ncols
  elif ncols == None:
    return nrows, np.ceil(min_cells / nrows)
  else:
    return nrows, ncols
