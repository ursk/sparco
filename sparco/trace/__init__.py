from IPython import embed

import logging
import os
import time

import traceutil.tracer

import sparco.sptools as sptools
import sparco.trace.sparse_coder
import sparco.trace.sp

def configure(conf):
  defaults = {
      'output_path': os.path.join(os.path.expanduser('~'), 'sparco_out'),
      'snapshot_interval': 100,
      'log_level': 'INFO'
      # 'profile_functions': { }  TODO implement later
      }
  settings = sptools.merge(defaults, conf)
  log_path = (settings.get('log_path')
      or os.path.join(settings['output_path'], 'sparco.log'))
  logging.basicConfig(filename=log_path,
      level=getattr(logging, settings['log_level'].upper()))
  sparco.trace.sp.Tracer.snapshot_interval = settings['snapshot_interval']
  trial_dir = os.path.join(settings['output_path'],
      'trial{0}'.format(time.strftime('%y%m%d%H%M%S')))
  sparco.trace.sparse_coder.Tracer.output_path = trial_dir
  traceutil.tracer.apply_tracer(sparco.trace.sp.Tracer, sparco.sp.RootSpikenet)
  traceutil.tracer.apply_tracer(sparco.trace.sparse_coder.Tracer,
      sparco.sparse_coder.SparseCoder)
