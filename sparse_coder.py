from sp import Spikenet

"""
Example calling pattern:

openmpirun -np 2 python run.py -m white -p white-12x12x8

"""

class SparseCoder:

  def __init__(self, configs):
		self.configs = configs

  def run(self):
		for config in self.configs:
			print 'Learning with lam = %g, maxit = %d, niter = %d' % (
					config['inference']['lam'], config['inference']['maxit'], config['niter'])
			sn = Spikenet(**config)
			sn.init_basis(file=file, phi=phi)
			sn.learn()
			phi = sn.phi.copy()
			eta = sn.learner.eta 
