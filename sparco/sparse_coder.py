import mpi
import sparco

class SparseCoder(object):

  def __init__(self, config):
    self.configs = config if isinstance(config, list) else [config]

  def run(self, phi=None, eta=.00001):
    self.phi, self.eta = phi, eta
    for self.t, config in enumerate(self.configs):
      config['phi'], config['eta'] = self.phi, self.eta
      self.iteration(config)

  def iteration(self, config):
    sn = self.create_spikenet(config)
    sn.run()
    self.phi, self.eta = sn.phi.copy(), sn.eta

  def create_spikenet(self, config):
    klass = sparco.RootSpikenet if mpi.rank == mpi.root else sparco.Spikenet
    return klass(**config)
