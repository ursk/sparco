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
import sparco
import sparco.sptools as sptools

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
      'db': None,
      'batch_size': 10,
      'num_iterations': 100,
      'phi': None,
      'inference_function': sparco.qn.sparseqn.sparseqn_batch,
      'inference_settings': {
        'lam': 0,
        'maxit': 15,
        'debug': False,
        'positive': False,
        'delta': 0.0001,
        'past': 6
        },
      'learner_class': sparco.learn.AngleChasingLearner,
      'eta': .00001,
      'eta_up_factor': 1.01,
      'eta_down_factor': .99,
      'target_angle': 1.,
      'max_angle': 2.,
      'basis_centering_interval': None,
      'basis_centering_max_shift': None,
      'writer_class': sparco.output.Writer,
      'write_interval': 100,
      'output_path': os.path.join(home, 'sn', 'py', 'spikes'),
      'create_plots': True,
      'logger_active': False,
      'log_path': os.path.join(home, 'sn', 'py', "vanilla.log"),
      'basis_method': 1,  # TODO this is a temporary measuer
      # 'logger_settings': {
      #   'echo': True,
      #   }
      }
    settings = sptools.merge(defaults, kwargs)
    for k,v in settings.items():
      setattr(self, k, v)
    self.learn_basis = getattr(self, "learn_basis{0}".format(self.basis_method))  # TODO temp

    self.phi /= sptools.vnorm(self.phi)
    sptools.mixin(self, self.learner_class)
    self.accumulated_basis_variance = np.zeros(self.N)
    self.validate_configuration()

  def validate_configuration(self):
    """Throw an exception for any invalid config parameter."""
    # if self.phi.shape != self.basis_dims:
    #   raise ValueError('Warm start phi wrong dimensions')
    if self.batch_size % mpi.procs != 0:
      raise ValueError('Batch size not multiple of number of procs')

### Learning
# methods here draw on methods provided by a learner mixin

  def run(self):
    """ Learn basis by alternative online minimization."""
    for self.t in range(self.num_iterations):
      self.phi = mpi.bcast(self.phi)
      self.iteration()

  def iteration(self):
    self.x = mpi.scatter(self.rootx)
    self.infer_coefficients()
    self.learn_basis()
    if self.t % self.update_coeff_statistics_interval == 0:
      self.update_coefficient_statistics()

  def infer_coefficients(self):
    self.a = self.inference_function(self.phi, self.x,
      **self.inference_settings)

  # more parallel, higher bandwidth requirement
  def learn_basis1(self):
    self.xhat = self.obj.compute_xhat(self.phi, self.a)
    self.dx = self.obj.compute_dx(self.phi, xhat=self.xhat)
    self.E = self.obj.compute_E(self.dx)
    self.dphi = self.obj.compute_dphi(self.dx, self.a)
    self.rootE = mpi.gather(self.E)
    self.rootdphi = mpi.gather(self.dphi)

  # less parallel, lower bandwidth requirement
  def learn_basis2(self):
    self.roota = mpi.gather(self.a, self.roota, mpi.root)

### Coefficient Statistics

  # TODO see if I can get the normalized norms in a single call
  def update_coefficient_statistics(self):
    self.a_l0norms = np.linalg.norm(self.a, ord=0, axis=1) / self.a.shape[1]
    self.root_a_l0norms = mpi.gather(self.a_l1norms)

    self.a_l1norms = np.linalg.norm(self.a, ord=0, axis=1)
    self.a_l1norms /= max(self.a_l1norms)
    self.root_a_l1norms = mpi.gather(self.a_l1norms)

    self.a_l2norms = np.linalg.norm(self.a, ord=0, axis=1)
    self.a_l2norms /= max(self.a_l1norms)
    self.root_a_l2norms = mpi.gather(self.a_l1norms)

    self.a_variance = np.var(self.a, axis=1)
    self.a_variance /= max(self.a_variance)
    self.root_a_variance = mpi.gather(self.a_l1norms)


class RootSpikenet(Spikenet):

  def __init__(self, **kwargs):
    super(RootSpikenet, self).__init__(**kwargs)
    sptools.mixin(self, self.writer_class)
    self.write_configuration(kwargs)
    if self.log_settings['active']:
      log_file = open(self.log_settings['path'], 'w+')
      sys.stdout = sparco.output.Logger(sys.stdout, log_file, **self.logger_settings)
      sys.stderr = sparco.output.Logger(sys.stderr, log_file, **self.logger_settings)

  def iteration(self):
    self.rootx = self.db.get_patches(self.batch_size)
    super(RootSpikenet, self).iteration()
    if self.write_interval and self.t % self.write_interval == 0:
      self.write_snapshot()

  @sptools.time_track
  def infer_coefficients(self):
    super(RootSpikenet, self).infer_coefficients()

  @sptools.time_track
  def learn_basis1(self):
    super(RootSpikenet, self).update_basis1()
    self.update_eta_and_phi()

  @sptools.time_track
  def learn_basis2(self):
    super(RootSpikenet, self).update_basis1()
    self.rootdx = np.array([self.compute_dx(self.rootA[i]) for i in range(self.batch_size)])
    self.rootE = np.array([self.compute_E(self.rootdx[i]) for i in range(self.batch_size)])
    self.rootdphi = np.array([self.compute_dphi(rootdx[i])] for i in range(self.batch_size))
    self.update_eta_and_phi(dphi)

  def update_eta_and_phi(self):
    self.meandphi = np.mean(self.rootdphi, axis=0)
    self.meanE = np.mean(self.rootE)
    self.proposed_phi = self.compute_proposed_phi(self.phi, self.meandphi, self.eta)
    angle = self.compute_angle(new_phi)
    self.update_phi()
    self.update_eta()

  def update_coefficient_statistics(self):
    super(RootSpikenet, self).update_coefficient_statistics()
    self.a_variance_cumulative += np.mean(self.root_a_variance, axis=0)
    self.basis_sort_order = np.argsort(self.accumulated_basis_variance)[::-1]
