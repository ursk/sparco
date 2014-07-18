import sparco.mpi as mpi
from traceutil.tracer import Tracer

# a configuration module needs to define a dictionary called `config`. This dictionary may contain up to three sub-dicitonaries under the keys `sampler`, `nets`, and `trace`.

config = {

    # Provides settings for the Sampler class. The Sampler 
    'sampler': {
      'cache_size': 1000, #T*subsample*cache  determines the batch size
      'resample': 2,
      'subsample': 2, # downsampling 128 1ms to 64 2ms
      'time_dimension': 0,
      'patch_length': 128
      },
    'nets': [],
    'trace': {
      'wrappers': {
        'RootSpikenet': {
          'learn_basis': [Tracer.t_log_execution_time],
          'infer_coefficients': [Tracer.t_log_execution_time],
          'load_patches': [Tracer.t_log_execution_time]
          }
        }
      }
    }

ladder = [[0.1, 5,  2000,  5.],
          [0.3, 10, 2000,  2.],
          [0.5, 20, 2000,  2.],
          [0.7, 25, 4000,  1.0],
          [0.9, 30, 10000, 0.5],
          [1.0, 35, 40000, 0.5]]

for lam, maxit, num_iterations, target_angle in ladder:
  config['nets'].append({
      'dictionary_size': 100,
      'convolution_time_length': 64,
      'batch_size': mpi.procs * 2,
      'num_iterations': num_iterations,
      'target_angle': target_angle,
      'max_angle': target_angle * 2,
      'inference_settings': {
        'lam': lam,
        'maxit': maxit
        }
      })
