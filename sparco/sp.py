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
    self.basis_dims = (self.C, self.N, self.P)
    self.coeff_dims = (self.N, self.T + self.P - 1)

    # init learner and normalize basis
    self.learner = self.learner_class(self.obj, self.dims,
        **self.learner_settings)
    self.phi /= sptools.vnorm(self.phi)

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
    # allocate MPI buffers
    if mpi.rank == mpi.root:
      self.A = np.empty((self.bs,) + self.coeff_dims)
      self.E = np.empty(self.bs)
      self.dphi = np.empty(self.bs + self.phi.shape)
    else:
      self.A = np.array([])
      self.E = np.empty([])
      self.dphi = np.empty([])

    self.X = np.array([])
    self.parX = np.empty((self.bs / mpi.procs, self.C, self.T)) 


    sparse_tic = 0
    learn_tic = 0

    # broadcast basis to nodes
    mpi.bcast(self.phi, mpi.root)

    for self.t in range(self.niter):

      self.select_data()
      mpi.scatter(self.X, self.parX, mpi.root)

      tic = time.time()
      self.infer_coefficients()
      sparse_tic += time.time() - tic

      # update basis in parallel
      tic = time.time()
      self.compute_error_and_dphi()
      self.learner.step(self.phi, self.A, self.X)
      learn_tic += time.time() - tic

      if mpi.rank == mpi.root and (self.t % self.disp) == 0:
        self.update_statistics()
        self.dump(self.A, self.X, self.t, sparse_tic, learn_tic)
        sparse_tic = learn_tic = 0

  def select_data(self):
    """Select random data patches on root and scatter to all nodes."""
    if mpi.rank == mpi.root:
      self.X = self.db.get_patches(self.bs)

  def infer_coefficients(self):
    """Compute the coefficients in parallel."""
    self.parA = self.inference_function(self.phi, self.parX,
      **self.inference_settings)
    # mpi.gather(parA, self.A, mpi.root)

  def learn_new_basis(self):
    self.parE, self.pardphi = self.learner_function(
        self.phi, self.parA, self.parX, self.T)
    mpi.gather(self.parE, self.E, mpi.root)
    mpi.gather(self.pardphi, self.dphi, mpi.root)
    if mpi.rank == mpi.root
      mean_dphi = np.mean(self.dphi, axis=0)
      new_phi = phi - (self.eta * mean_dphi)
      new_phi /= sptools.vnorm(dphi)




  def dump(self, A, X, t, sparse_tic, learn_tic):
    """
    Dump iteration data to files
    """
    # l0 norm of all coefficients across batches
    l0norm = (self.A != 0.).sum().astype(np.float64) / np.prod(self.A.shape)

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

  def update_statistics(self):
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
    if self.variance is None:
        self.variance = variance
    else:
        self.variance += variance
    variance /= max(variance)
    order = np.argsort(self.variance)[::-1]


  def obj(self, phi, A, X, l1=False, recon=False):
    """Compute the objective function.

    Args:
      phi  - basis
      A    - coefficients  (batch, basis, time)
      X    - data (batch, channel, time)
      l1   - add L1 of A to objective if True
      recon  - return objective, derivative, reconstructed batches
    """
    dphi = np.zeros_like(phi)
    E = 0.
    xhat = np.zeros_like(X)
    for pat in range(self.bs):
      for b in range(self.P):
        print "pat: {0}".format(pat)
        print "b: {0}".format(b)
        print "Phi shape: {0}".format(phi.shape)
        print "A shape: {0}".format(A.shape)
        xhat[pat] += np.dot(phi[:,:,b], A[pat,:,b:b+self.T])

      dx = xhat[pat] - X[pat]
      E += 0.5 * np.linalg.norm(dx)**2

      for b in range(self.P):
        dphi[:,:,b] += np.dot(dx, A[pat,:,b:b+self.T].T)

    E /= self.bs
    dphi /= self.bs

    if l1:
      E += self.inference_settings['lam'] * abs(A).sum() / self.bs

    if recon:
      return E, dphi, xhat
    else:
      return E, dphi

  def obj(self, phi):
