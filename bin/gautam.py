#!/usr/bin/env python

from IPython import embed

import argparse
import glob
import os
import sys

from mpi4py import MPI

sys.path.append(
		os.path.normpath(os.path.join(os.path.realpath(__file__), '..', '..')))
sys.path.append(
		os.path.normpath(os.path.join(os.path.realpath(__file__), '..', '..', 'qn')))

from datadb import DB
from sparse_coder import SparseCoder

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-directory',
		help='path to directory containing input files')
parser.add_argument('-o', '--output-directory',
		help='path to directory containing output files')
args = parser.parse_args()

channels = range(64)
dims = (len(channels), 100, 64, 2*64)

db = DB(**{
		'dims': dims,
		'channels': range(64),
		'filenames': glob.glob(os.path.join(args.input_directory, '*.h5')),
		'cache': 50, #T*subsample*cache  determines the batch size
		'resample': 2,
		'cull': 0.,
		'maxcull': 10., # (URS) changed 5 to 10 because a lot of times patches were rejected. Changed back: This is a problem with the data being too white
		'std_threshold': 0., # default was 2 but does not work with climate data?
		'subsample': 2, # downsampling 128 1ms to 64 2ms
		'normalize': 'patch',
		'smooth': False,
		'line': False,
		'Fs': 1000
		})

ladder = [[0.1,	 5,	 2000, 5.],
					[0.3, 10,	 2000, 2.],
					[0.5, 20,	 2000, 2.],
					[0.7, 25,	 4000, 1.0],
					[0.9, 30, 10000, 0.5],
					[1.0, 35, 40000, 0.5]]
configs = []

# embed()

for lam, maxit, niter, target in ladder:
	configs.append({
		  # dims: (num channels, num basis funcs, shift, time steps per basis func)
			'db': db,
			'dims': dims,
			'bs': MPI.COMM_WORLD.Get_size() * 2,
			'niter': 10000,
			'niter': niter,
			'learner': {
				'eta': 0.0001,
				'target': target,
				'thresh': target*2
				},
			'inference': {
				'lam': lam,
				'maxit': maxit
				},
			'writer': {
				'output_path': args.output_directory,
				'prefix': 'gautam',
				}
			})
sparse_coder = SparseCoder(configs)
sparse_coder.run()
