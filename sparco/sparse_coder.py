from IPython import embed
import os.path

import numpy as np

# import sparco.mpi as mpi
import mpi
import sparco
import sptools

"""
Example calling pattern:

openmpirun -np 2 python run.py -m white -p white-12x12x8

"""

class SparseCoder:

  def __init__(self, configs, output_path):
    self.configs = configs
    self.output_path = output_path

  def run(self, basis_dims=None, phi=None, eta=.00001):
    if mpi.rank == mpi.root:
      phi = phi or self.generate_random_basis(basis_dims)
      phi /= sptools.vnorm(self.phi)
    for i, config in enumerate(self.configs):
      mpi.bcast(self.phi)
      config_tuple = (i, config['num_iterations'],
        config['inference_settings']['lam'],
        config['inference_settings']['maxit'])
      self.log_status(config_tuple)
      klass = sparco.RootSpikenet if mpi.rank == mpi.root else sparco.Spikenet
      sn = klass(eta=eta, phi=phi,
          output_path=self.get_inner_output_path(config_tuple), **config)
      sn.run()
      phi = sn.phi.copy()
      eta = sn.eta 

  def log_status(self, config_tuple):
    print 'Round %d: num_iterations = %d, lam = %g, maxit = %d' % config_tuple
    
  def get_inner_output_path(self, config_tuple):
    dir = "{0}_niter_{1}_lam_{2}_maxit_{3}".format(*config_tuple)
    return os.path.join(self.output_path, dir)

  def generate_random_basis(self, dims, filter=False, correlated=False):
    """Generate a random basis matrix.

    Args:
      filter  - filter random normal
      correlated - ????
    """
    if correlated:  # vertical stripes + random pixel noise
      phi = np.empty(dims)
      phi[:,:] = np.random.randn(
          self.dims[1], dims[2])[None,:]
      phi += 0.5*np.random.randn(*dims)
    else:
      phi = np.random.randn(*dims)
    if filter:
      phi = sptools.blur(self.phi)
    return phi
