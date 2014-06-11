import numpy as np

import mpi
from sp import Spikenet

"""
Example calling pattern:

openmpirun -np 2 python run.py -m white -p white-12x12x8

"""

class SparseCoder:

  def __init__(self, configs):
    self.configs = configs

  def run(self):
    phi = None
    eta = self.configs[0]['learner_settings']['eta']
    basis_dims = [self.configs[0][k] for k in ('C','N','P')]
    for config in self.configs:
      print 'Learning with lam = %g, maxit = %d, niter = %d' % (
          config['inference_settings']['lam'],
          config['inference_settings']['maxit'], config['niter'])
      if mpi.rank == mpi.root:
        config['phi'] = (phi or self.generate_random_basis(basis_dims))
      else:
        config['phi'] = np.empty(basis_dims)
      config['learner_settings']['eta'] = eta
      sn = Spikenet(**config)
      sn.learn()
      phi = sn.phi.copy()
      eta = sn.learner.eta 

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
