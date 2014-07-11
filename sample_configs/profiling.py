import copy
import itertools
import os
import time

import numpy as np
import pfacets
from traceutil.tracer import Tracer

import sparco.mpi as mpi
import sparco.sampler

config = {
    'run_ladder': False,
    'sampler': {
      'cache_size': 1000, #T*subsample*cache  determines the batch size
      'resample': 2,
      'subsample': 2, # downsampling 128 1ms to 64 2ms
      'time_dimension': 0,
      'patch_length': 128
      },
    'nets': [],
    'trace': {
      'RootSpikenet': {
        'wrappers': {
          'run': [Tracer.t_dump_data],
          'learn_basis': [Tracer.t_log_execution_time],
          'infer_coefficients': [Tracer.t_log_execution_time],
          'load_patches': [Tracer.t_log_execution_time]
          }
        },
      'SparseCoder': {
        'create_spikenet_directory': False
        }
      }
    }

profile_space = {
    'channels': [16,32,64],
    'patch_length': [128],
    'patches_per_node': [1,2,4],
    'basis_method': [1,2]
    }
profile_configs = map(lambda x: dict(zip(profile_space.keys(), x)),
    itertools.product(*tuple(profile_space.values())))

trial_base_dir = 'profiling{0}'.format(time.strftime('%y%m%d%H%M%S'))

for pc in profile_configs:
  dirname = '_'.join(itertools.imap(
    lambda k,v: '{0}_{1}'.format(k,v), pc.keys(), pc.values()))
  config['nets'].append({
    'trace': {
      # 'trial_directory': os.path.join(trial_base_dir, dirname)
      'trial_directory': dirname
      },
    'sampler': pfacets.merge( copy.deepcopy(config['sampler']),
      {
        'channels': np.arange(pc['channels']),
        'patches_length': pc['patch_length']
        }
      ),
    'basis_method': pc['basis_method'],
    'dictionary_size': 100,
    'convolution_time_length': 64,
    'batch_size': mpi.procs * pc['patches_per_node'],
    'num_iterations': 2000,
    'run_time_limit': 120,
    'target_angle': 5,
    'max_angle': 10,
    'inference_settings': {
      'lam': 0.1,
      'maxit': 10
      }
    })
