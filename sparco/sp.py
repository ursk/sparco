"""
A convolution based sparsenet of an array of time series

Key:
 phi - basis [channel, neuron, coefficients]
 A   - coefficients
 X   - patches of data

Note:
1. The kernels are convolution kernels.
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import mpi
import sparseqn
import sptools
from logger import Logger
from basis_writer import Writer
import learner

class Spikenet(object):
  """
  Settings:
    db: DB. Data source.
    phi: Numpy Array. Must be CxNxP. The initial basis matrix.
    disp: Integer. Iteration interval at which to write basis
    Dimensions:
      C:  Integer. Number channels
      N:  Integer. Number kernels
      P:  Integer. Time points in convolution kernel
      T:  Integer. Time points in raw data (T >> P)
    Inference and learning:
      niter: Integer. Number of iterations for learning
      bs: Integer. Batch size to be divided among procs.
      inference_function: Function.
      inference_settings: Dict. Kwargs for inference_function.
      learner_class: Class. Must implement step(phi, A, X)
      learner_settings: Keyword arguments for learner class constructor.
    Output:
      logger_active:  Boolean. Enable logging.
      log_path: String. Path to log file.
      logger_settings: Keyword arguments for logger initializer.
      echo:  echo stdout to console
      movie:  use movie basis writer (for natural scenes)
      prefix:  prepend to output file names
      plots:  whether or not to output plots (due to memory leak in matplotlib)
  """

  def __init__(self, **kwargs):
    """Set and validate configuration, initialize output classes."""
    home = os.path.expanduser('~')
    defaults = {
      'C' : None,
      'N' : None,
      'P' : None,
      'T' : None,
      'db': None,
      'phi': None,
      'bs': 10,
      'niter': 100,
      'disp': 100,
      'plots': True,
      'inference_function': sparseqn.sparseqn_batch,
      'inference_settings': {
        'lam': 0,
        'maxit': 15,
        'debug': False,
        'positive': False,
        'delta': 0.0001,
        'past': 6
        },
      'learner_class': learner.AngleChasing,
      'learner_settings': {
        'eta': .00001,
        'up': 1.01,
        'down': .99,
        'target': 1.,
        'thresh': 2.,
        'c': 500,
        'cmax': 1,
        'smooth': False
        },
      'writer_settings':{
        'output_path': os.path.join(home, 'sn', 'py', 'spikes'),
        'prefix': 'vanilla',
        'is_movie': False,
        'create_plots': True,
        },
      # 'log': None
      # }
      'logger_active': False,
      'log_path': os.path.join(home, 'sn', 'py', "vanilla.log"),
      'logger_settings': {
        'echo': True,
        }
      }
    settings = sptools.merge(defaults, kwargs)
    for k,v in settings.items():
      setattr(self, k, v)

    # set dimensions
    self.dims = (self.C, self.N, self.P, self.T)
    self.patch_dims = (self.C, self.T)
    self.basis_dims = (self.C, self.N, self.P)
    self.coeff_dims = (self.N, self.T + self.P - 1)

    # init learner and normalize basis
    self.learner = self.learner_class(self.obj, self.dims,
        **self.learner_settings)
    self.phi /= sptools.vnorm(self.phi)

    self.accumulated_basis_variance = np.zeros(self.N)

    self.validate_configuration()

    # initialize logging and output
    if self.logger_active:
      log_file = open(config['log_path'] , 'w+', 0)
      sys.stdout = Logger(sys.stdout, log_file, **self.logger_settings)
      sys.stderr = Logger(sys.stderr, log_file, **self.logger_settings)
    self.writer = Writer(**self.writer_settings)
    if mpi.rank == mpi.root:
      self.writer = Writer(**self.writer_settings)
      self.writer.write_configuration(settings)

  def validate_configuration(self):
    """Throw an exception for any invalid config parameter."""
    if self.phi.shape != self.basis_dims:
      raise ValueError('Warm start phi wrong dimensions')
    if self.bs % mpi.procs != 0:
      raise ValueError('Batch size not multiple of number of procs')

  def learn(self):
    """ Learn basis by alternative online minimization."""
    mpi.bcast(self.phi, mpi.root)
    # self.allocate_buffers()
    for self.t in range(self.niter):
      self.iteration()
      # self.load_data()
      # self.infer_coefficients()
      # self.update_basis()
      # if self.t % self.write_interval == 0:
      #   self.update_statistics()

  def iteration(self):
    self.x = mpi.scatter(self.rootx)
    self.infer_coefficients()
    self.update_basis()

  # def allocate_buffers(self):
  #   self.X = np.empty((self.bs / mpi.procs,) + self.patch_dims) 
  #   self.A = coeff_dims
    # if mpi.rank == mpi.root:
    #   self.rootX = None  # filled in select_data()
    #   self.rootA = np.empty((self.bs,) + self.coeff_dims)
    #   self.rootE = np.empty(self.bs)
    #   self.rootdphi = np.empty(self.bs + self.phi.shape)

  # @time_track
  # def load_data(self):
  #   """Select random data patches on root and scatter to all nodes."""
    # if mpi.rank == mpi.root:
    #   self.rootX = self.db.get_patches(self.bs)
    # mpi.scatter(self.rootX, self.X, mpi.root)

  # @time_track
  def infer_coefficients(self):
    """Compute the coefficients."""
    self.a = self.inference_function(self.phi, self.x,
      **self.inference_settings)
    # mpi.gather(parA, self.A, mpi.root)

### Learning
# methods here draw on methods provided by a learner mixin

  # more parallel, higher bandwidth requirement
  def update_basis1(self):
    self.xhat = self.obj.compute_xhat(self.phi, self.a)
    self.dx = self.obj.compute_dx(self.phi, xhat=self.xhat)
    self.E = self.obj.compute_E(self.dx)
    self.dphi = self.obj.compute_dphi(self.dx, self.a)
    self.rootE = mpi.gather(self.E)
    self.rootdphi = mpi.gather(self.dphi)

  # less parallel, lower bandwidth requirement
  def update_basis2(self):
    mpi.gather(self.A, self.rootA, mpi.root)

### Descriptive Statistics

  def compute_coefficient_statistics(self, A):
    """Compute l0, l1, l2 norms and variance for each basis functions coeffs.
    """
    coeff = A.transpose(1,0,2).copy()
    coeff.shape = (coeff.shape[0], coeff.shape[1]*coeff.shape[2])
    
    # get norms and variance of coefficients
    l0norm = (coeff != 0.).sum(axis=1).astype('float64')
    l0norm /= np.prod(coeff.shape[1:])
    l0 = np.mean(l0norm)

    l1norm = np.abs(coeff).sum(axis=1)
    l1norm /= max(l1norm)

    l2norm = norm(phi)
    l2norm /= max(l2norm)

    # sort based on accumulated variance but display recent batch variance
    variance = np.var(coeff, axis=1)
    self.accumulated_basis_variance += variance

    variance /= max(variance)
    order = np.argsort(self.variance)[::-1]

### Output

  def dump(self, A, X, t, sparse_tic, learn_tic):
    """
    Dump iteration data to files
    """
    # l0 norm of all coefficients across batches
    # l0norm = (self.A != 0.).sum().astype(np.float64) / np.prod(self.A.shape)

    # normalized reconstruction error
    E, gd, Xhat = self.obj(self.phi, self.A, self.X, recon=True, l1=False)
    z = self.obj(self.phi, np.zeros_like(self.A), self.X)[0]
    error = E / z
    snr = 10 * np.log10( 1 / error )

    out = '[%d] Inference: %.4fs, Learning: %.4fs, L0: %.8f, E: %f, SNR: %f'
    print out % (t, sparse_tic, learn_tic, l0norm, error, snr)

    # write basis to file
    # TODO clean up this call by passing just the SPikenet
    self.basis_writer.write(self.phi, A, t,#self.code(t),
                error=error, X=X, Xhat=Xhat, spikenet=self, t=t)

### Profiling

  PROFILING_TABLE = {}
  def time_track(orig):
    PROFILING_TABLE[orig.__name__] = []
    def tracked_function(*args, **kwargs):
      start = time.now()
      res = orig(*args, **kwargs)
      end = time.now()
      PROFILING_TABLE[orig.__name__].append(start - end)
      return res
    return tracked_function if mpi.rank == mpi.root else orig

  # def obj(self, phi, A, X, l1=False, recon=False):
  #   """Compute the objective function.
  #
  #   Args:
  #     phi  - basis
  #     A    - coefficients  (batch, basis, time)
  #     X    - data (batch, channel, time)
  #     l1   - add L1 of A to objective if True
  #     recon  - return objective, derivative, reconstructed batches
  #   """
  #   dphi = np.zeros_like(phi)
  #   E = 0.
  #   xhat = np.zeros_like(X)
  #   for pat in range(self.bs):
  #     for b in range(self.P):
  #       print "pat: {0}".format(pat)
  #       print "b: {0}".format(b)
  #       print "Phi shape: {0}".format(phi.shape)
  #       print "A shape: {0}".format(A.shape)
  #       xhat[pat] += np.dot(phi[:,:,b], A[pat,:,b:b+self.T])
  #
  #     dx = xhat[pat] - X[pat]
  #     E += 0.5 * np.linalg.norm(dx)**2
  #
  #     for b in range(self.P):
  #       dphi[:,:,b] += np.dot(dx, A[pat,:,b:b+self.T].T)
  #
  #   E /= self.bs
  #   dphi /= self.bs
  #
  #   if l1:
  #     E += self.inference_settings['lam'] * abs(A).sum() / self.bs
  #
  #   if recon:
  #     return E, dphi, xhat
  #   else:
  #     return E, dphi
