#!/usr/bin/env python

from IPython import embed

import argparse
import itertools
import time
import glob
import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.realpath(__file__), '..', '..')))
# TODO fix this-- the issue is that tokyo is not on the load path
sys.path.append(
    os.path.normpath(os.path.join(os.path.realpath(__file__), '..', '..', 'sparco', 'qn')))

import sparco
import sparco.cli
import sparco.mpi as mpi
# from datadb import DB
# from sparse_coder import SparseCoder

parser = sparco.cli.arg_parser
parser.add_argument('-I', '--inner-directory', action='store_true',
    help='do not create a directory inside the output directory')
args = parser.parse_args()

C = 64  # num channels
N = 100  # num basis functions
P = 64  # time steps per basis function
T = 128  # time size of 

def make_db(channels):
  return  sparco.db.DB(**{
    'dims': (C, N, P, T),
    'channels': range(channels),
    'filenames': glob.glob(os.path.join(args.input_directory, '*.h5')),
    'cache': 50, #T*subsample*cache  determines the batch size
    'resample': 2,
    'cull': 0.,
    'maxcull': 5., # (URS) changed 5 to 10 because a lot of times patches were rejected. Changed back: This is a problem with the data being too white
    'std_threshold': 0., # default was 2 but does not work with climate data?
    'subsample': 2, # downsampling 128 1ms to 64 2ms
    'normalize': 'patch',
    'smooth': False,
    'line': False,
    'Fs': 1000
    })


profile_space = {
    # 'channels': [16,32,64],
    'channels': [64],
    'patch_size': [128],
    'patches_per_node': [1,2,4],
    # 'nodes': ????,
    'basis_method': [1,2]
    # 'basis_method': [2]
    }
profile_configs = map(lambda x: dict(zip(profile_space.keys(), x)),
    itertools.product(*tuple(profile_space.values())))

if args.inner_directory:
  base_path = os.path.join(args.output_directory,
      'profiling{0}'.format(time.strftime('%y%m%d%H%M%S')))
else:
  base_path = args.output_directory

for pc in profile_configs:

  output_path = os.path.join(base_path,
      '_'.join(map(lambda p: '{0}_{1}'.format(p[0],p[1]), pc.items())))
  db = make_db(pc['channels'])

  configs = [{
      'db': db,
      'T': pc['patch_size'],
      'batch_size': mpi.procs * pc['patches_per_node'],
      'num_iterations': 2000,
      'run_time_limit': 120,
      'target_angle': 5,
      'max_angle': 10,
      'create_plots': False,
      'basis_method' : pc['basis_method'],
      'inference_settings': {
        'lam': 0.1,
        'maxit': 5
        },
      }]
  sparse_coder = sparco.sparse_coder.SparseCoder(configs, output_path)
  sparse_coder.run(basis_dims=(pc['channels'],N,P), eta=.00001)
