"""
Run unsupervised algorithm on data
	
Several different test cases:
1. whitened natural movies
2. polytrode data filtered 500Hz-10kHz
3. polytrode data filtered 0-150Hz
4. annie semichronic data
5. sparse morlet basis
	
"""
import mpi4py.rc
mpi4py.rc.profile('MPE')
from mpi4py import MPI # Anaconda dies here! 
	
import matplotlib
matplotlib.use('Agg')
matplotlib.interactive(False)
	
import numpy as np
import sp
from sp import Spikenet
import scipy
import matplotlib.pyplot as plt
import datadb
reload(datadb)
import learner
reload(learner)
#import sparse # different cythoned library, probably won't need it?
#import mp # not used here
# import blbfgs # not used here 
#import sptools
import os, sys
import ipdb
import h5py
	
home = os.path.expanduser('~')
path1 = os.path.join(home, 'Dropbox/nersc/csc/spikes/qn') # Cerberus
path2 = os.path.join(home, 'csc/spikes/qn')				  # Hopper
if not path1 in sys.path: sys.path.append(path1)
if not path2 in sys.path: sys.path.append(path2)
print "pathes", path1, path2
import sparseqn	 
	
rank = MPI.COMM_WORLD.Get_rank()
procs = MPI.COMM_WORLD.Get_size()
root = 0
mpi = (rank, procs, root)
logger = False
	
print 'Rank = %d, procs = %d, root = %d' % mpi

def gautam(file=None, prefix='gautam'):
	"""
	(URS) Dummy climate data to test the toolbox
	based on lowpass
	"""
	# choose channels
	channels = range(64)
	print 'Using channels: ', channels

	C = len(channels)
	N = 100 # basis functions
	P = 64 # maybe this is the amount of shift?
	T = 2*P # 128 time steps in one basis

	dims = (C, N, P, T)

	bs = procs*2 # scaling BatchSize with number of MPI threads
	niter = 10000
	disp = 100
	plots = True

	# inference
	# positive l1 quasinewton -- URS removed MP option
	if True:
		inference = sparseqn.sparseqn_batch
		lam = .2
		iargs = {'lam': lam, 'maxit':15, 'debug':False, 'positive':False,  ## Set True for nonnegative coefficients
				 'delta':0.0001, 'past':6}

	# learning
	if True:
		method = learner.AngleChasing # [URS] thresh is ignored, set to target*2
		largs = {'eta': .00001, 'up': 1.01, 'down': .99, 'target': 1., 'thresh': 2.,
				 'mpi': mpi, 'c': 500, 'cmax': 1, 'smooth': False}


	# initialize data database
	db = None
	if rank == root:
		
		# choose data files
		home = os.path.expanduser('~')
		basedir = os.path.join(home, 'Dropbox', 'nersc', 'data')
		#basedir = os.path.join('/project/projectdirs/m636/neuro/polytrode/csc', 'data')
		files =range(1)
		#filenames = ['full_climate_testdata%d.h5' % (i) for i in files]	
		filenames = ['gautam_testdata%d.h5' % (i) for i in files]
		full = [os.path.join(basedir, f) for f in filenames]
		filenames = []
		for f in full:
			if os.path.exists(f): filenames.append(f)

			
		kwargs = {'channels': channels,
				  'filenames': filenames,
				  'cache': 50, #T*subsample*cache  determines the batch size
				  'resample': 2,
				  'cull': 0.,
				  'maxcull': 10., # (URS) changed 5 to 10 because a lot of times patches were rejected. Changed back: This is a problem with the data being too white
				  'std_threshold': 0., # default was 2 but does not work with climate data?
				  'subsample': 2, # downsampling 128 1ms to 64 2ms
				  'normalize': 'patch',
				  'smooth': False,
				  'line': False,
				  'Fs': 1000}
				  
		db = datadb.ClimateDB(dims, **kwargs)
		
	p = {'db': db,
		 'bs': bs,
		 'niter': niter,
		 'disp': disp,
		 'mpi': mpi,
		 'learner': method,
		 'largs': largs,
		 'inference': inference,
		 'iargs': iargs,
		 'logger': logger,
		 'prefix': prefix,
		 'plots': plots}
		 
	# create annealing schedule of lam, maxit, niter, target
	if False:
		ladder = []
		lam = 0.1
		linc = 0.1
		maxit = 5
		minc = 1
		for i in range(20):
			ladder.append([lam, maxit, 1000])
			lam += linc
			maxit += minc

	# Values for 64 x 128 basis 
	#	   (lam, maxit, niter, target)
	ladder = [[0.1,	 5,	 2000, 5.], # [URS] upped target from 2 to 5 so thresh is high enough (thresh=target*2)
			  [0.3, 10,	 2000, 2.],
			  [0.5, 20,	 2000, 2.],
			  [0.7, 25,	 4000, 1.0],
			  [0.9, 30, 10000, 0.5],
			  [1.0, 35, 40000, 0.5]]
              
	phi = np.random.randn(C,N,P) # random init for sparse coefs: 35 channels x 35 basis fns x 64 ???
	eta = 0.0001
    
	# h5 = h5py.File('/home/amir/sn/py/spikes/out/tp6-low-3/basis.h5')
	# oldphi = h5['phi'][:]
	# phi[:,:oldphi.shape[1]] = oldphi
    
	for lam, maxit, niter, target in ladder:
		print 'Learning with lam = %g, maxit = %d, niter = %d' % (lam, maxit, niter)
		iargs['lam'] = lam
		iargs['maxit'] = maxit
		largs['eta'] = eta
		largs['target'] = target
		largs['thresh'] = target*2
		p['niter'] = niter
		sn = Spikenet(dims, **p)
		sn.init_basis(file=file, phi=phi)
		sn.learn()
		phi = sn.phi.copy()
		eta = sn.learner.eta 



def ecog(file=None, prefix='ecog'):
	"""
	(KRIS) Egoc data
	based on lowpass
	"""
	# choose channels
	import scipy.io
	a=scipy.io.loadmat('/Users/urs/Dropbox_outsource/nersc/data/EC2.cntrlslc.cplngchns.mat')
	channels = a['elects1'].flatten()-1
	#channels = range(256)
	print 'Using channels: ', channels

	C = len(channels)
	N = 20 # basis functions
	P = 64 # maybe this is the amount of shift?
	T = 2*P # 128 time steps in one basis

	dims = (C, N, P, T)

	bs = procs*2 # scaling BatchSize with number of MPI threads
	niter = 10000
	disp = 100
	plots = True

	# inference
	# positive l1 quasinewton -- URS removed MP option
	if True:
		inference = sparseqn.sparseqn_batch
		lam = .2
		iargs = {'lam': lam, 'maxit':15, 'debug':False, 'positive':False,  ## Set Trtue for nonnegative coefficients
				 'delta':0.0001, 'past':6}

	# learning
	if True:
		method = learner.AngleChasing # [URS] thresh is ignored, set to target*2
		largs = {'eta': .00001, 'up': 1.01, 'down': .99, 'target': 1., 'thresh': 2.,
				 'mpi': mpi, 'c': 500, 'cmax': 1, 'smooth': False}


	# initialize data database
	db = None
	if rank == root:

		# choose data files
		home = os.path.expanduser('~')
		basedir = os.path.join(home, 'Dropbox_outsource', 'nersc', 'data', 'EC2_CV_trials')
		#basedir = os.path.join('/project/projectdirs/m636/neuro/polytrode/csc', 'data')
		files = range(1,50)
		filenames = ['EC2_CV_trl%d.h5' % (i) for i in files]
		#filenames = ['EC2_CV.nrl5']
		full = [os.path.join(basedir, f) for f in filenames]
		
		filenames = []
		for f in full:
			if os.path.exists(f): filenames.append(f)
        
		kwargs = {'channels': channels,
				  'filenames': filenames,
				  'cache': 2, #T*subsample*cache  determines the batch size
				  'resample': 1,
				  'cull': 0.,
				  'maxcull': 100., # (URS) changed 5 to 10 because a lot of times patches were rejected. Changed back: This is a problem with the data being too white
				  'std_threshold': 0., # default was 2 but does not work with climate data?
				  'subsample': 1, # downsampling 128 1ms to 64 2ms
				  'normalize': 'patch',
				  'smooth': False,
				  'line': False,
				  'Fs': 200}

		db = datadb.EcogDB(dims, **kwargs)

	p = {'db': db,
		 'bs': bs,
		 'niter': niter,
		 'disp': disp,
		 'mpi': mpi,
		 'learner': method,
		 'largs': largs,
		 'inference': inference,
		 'iargs': iargs,
		 'logger': logger,
		 'prefix': prefix,
		 'plots': plots}

	# create annealing schedule of lam, maxit, niter, target
	if False:
		ladder = []
		lam = 0.1
		linc = 0.1
		maxit = 5
		minc = 1
		for i in range(20):
			ladder.append([lam, maxit, 1000])
			lam += linc
			maxit += minc

	# Values for 64 x 128 basis 
	#	   (lam, maxit, niter, target)
	ladder = [[0.1,	 5,	 2000, 5.], # [URS] upped target from 2 to 5 so thresh is high enough (thresh=target*2)
			  [0.3, 10,	 2000, 2.],
			  [0.5, 20,	 2000, 2.],
			  [0.7, 25,	 4000, 1.0],
			  [0.9, 30, 10000, 0.5],
			  [1.0, 35, 40000, 0.5]]

	phi = np.random.randn(C,N,P) # random init for sparse coefs: 35 channels x 35 basis fns x 64 ???
	eta = 0.00001

	# h5 = h5py.File('/home/amir/sn/py/spikes/out/tp6-low-3/basis.h5')
	# oldphi = h5['phi'][:]
	# phi[:,:oldphi.shape[1]] = oldphi

	for lam, maxit, niter, target in ladder:
		print 'Learning with lam = %g, maxit = %d, niter = %d' % (lam, maxit, niter)
		iargs['lam'] = lam
		iargs['maxit'] = maxit
		largs['eta'] = eta
		largs['target'] = target
		largs['thresh'] = target*2
		p['niter'] = niter
		sn = Spikenet(dims, **p)
		sn.init_basis(file=file, phi=phi)
		sn.learn()
		phi = sn.phi.copy()
		eta = sn.learner.eta 




if __name__ == '__main__':
	"""
	Example calling pattern:
	
	openmpirun -np 2 python run.py -m white -p white-12x12x8
	
	"""
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-m', '--model', dest='model', help='specifies model')
	parser.add_option('-w', '--warm', dest='warm', help='warm start')
	parser.add_option('-p', '--prefix', dest='prefix', help='output prefix')
	parser.add_option('-r', '--recording', dest='recording', help='recording session, eg. tiger/p6')
	parser.add_option('-s', '--session', dest='session', help='recording session, eg. 1')			 
	(options, args) = parser.parse_args()
	file = options.warm
	prefix = options.prefix
	recording = options.recording
	if options.session is not None:
		session = int(options.session)
	
	if options.model is None:
		white(file=options.warm, prefix='white')
	elif options.model == 'gautam': # URS: new addition for testing
		gautam(file, prefix)	 
	elif options.model == 'ecog': # 
		ecog(file, prefix)	  
	else:
		raise ValueError('Invalid parameters passed.')



	
"""
ToDo: visualize Gautams LFP data:
- Chunk into trials defined by the "tr" variable (88 trials)
- use place variable "pl" discretized into 100 linear places 
- build a 2xT vector of trial and place.
- Make trial vs. position raster plots for each of the basis functions. Location tuning?

(this might be interesting for V1 data too, look for receptive fields? Could be good for a 
paper if we show on white noise that gamma power is too slow to resolve the RF. One wave of
50Hz is 20ms long, noise frames are 50ms, so probably not. ) 

Call to run this locally:
  /opt/local/bin/openmpirun -np 3 /opt/local/bin/python run.py -m climate -p gautam_pca_whitened
since macports is the only python that understands MPI. No more!   
  /Users/urs/anaconda/bin/mpirun -np 3 /Users/urs/anaconda/bin/python run.py -m climate -p gautam_pca_whitened
actually works now!

Framework:
run.py 
  db = datadb.ClimateDB(dims, **kwargs)
  db is packaged into p
  sn = Spikenet(dims, **p) # in sp.py
  sn.learn()



"""
