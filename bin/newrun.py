import mpi4py.rc
mpi4py.rc.profile('MPE')
from mpi4py import MPI # Anaconda dies here! 
from sp import Spikenet

class SparseCoder:

  def __init__(self, configs):
    rank = MPI.COMM_WORLD.Get_rank()
    procs = MPI.COMM_WORLD.Get_size()
    root = 0
    defaults = {
			'mpi': (rank, procs, root),
      'dims': None,
      'db': None,
      'bs': procs*2,
      'niter': 10000,
      'disp': 100,
      'plots': True,
      'inference_class': sparseqn.sparseqn_batch,
      'lam': .2,
      'inference': {
        'lam': .2,
        'maxit': 15,
        'debug': False,
        'positive': False,
        'delta': 0.0001,
        'past': 6
        },
      'learner_class': learner.AngleChasing,
      'learner': {
        'eta': .00001,
        'up': 1.01,
        'down': .99,
        'target': 1.,
        'thresh': 2.,
        'mpi': (rank, procs, root),
        'c': 500,
        'cmax': 1,
        'smooth': False
        },
      'writer':{
        'output_path': os.path.join(home, 'sn', 'py', 'spikes'),
        'prefix': 'sparco',
        'movie': False,
        'plots': True,
        },
      'logger_active': False,
			'log_path': os.path.join(output_path, "{0}.log".format(prefix)),
      'logger': {
        'echo': True,
        'rank': rank,
        }
      }
    self.ladder = map(lambda x: copy.copy(defaults).merge(x), configs)

  def run(self):
		for config in self.ladder:
			print 'Learning with lam = %g, maxit = %d, niter = %d' % (
					config['lam'], config['inference']['maxit'], config['niter'])
			sn = Spikenet(dims, **p)
			sn.init_basis(file=file, phi=phi)
			sn.learn()
			phi = sn.phi.copy()
			eta = sn.learner.eta 
