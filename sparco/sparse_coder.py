import numpy as np

import mpi
from sp import Spikenet

"""
Example calling pattern:

openmpirun -np 2 python run.py -m white -p white-12x12x8

"""

class SparseCoder:

  def __init__(self, configs, output_path):
    self.configs = configs
    self.output_path = output_path

  def run(self):
    phi = None
    eta = self.configs[0]['learner_settings']['eta']
    basis_dims = [self.configs[0][k] for k in ('C','N','P')]
    for i, config in enumerate(self.configs):
      iteration_tuple = (config['inference_settings']['lam'],
          config['inference_settings']['maxit'], config['niter'])
      print 'Learning with lam = %g, maxit = %d, niter = %d' % iteration_tuple
      config['writer_settings']['output_path'] = os.path.join(
          self.output_path, "%d_lam_%g_maxit_%d_niter_%d".format(*iteration_tuple))
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
