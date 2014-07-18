import logging
import os
import time

import pfacets
import traceutil.tracer

import sparco.trace.sparse_coder
import sparco.trace.sp

def configure(**kwargs):
  defaults = {
      'coder': None,
      'spikenet': None,
      'output_root': os.path.join(os.path.expanduser('~'), 'sparco_out'),
      'trial_directory': 'trial{0}'.format(time.strftime('%y%m%d%H%M%S')),
      'snapshot_interval': 100,
      'log_level': 'INFO',
      'log_format': '%(asctime)s %(message)s',
      'RootSpikenet': {},
      'SparseCoder': {}
      }
  settings = pfacets.merge(defaults, kwargs)
  settings['SparseCoder']['RootSpikenet_config'] = settings['RootSpikenet']

  output_path = os.path.join(settings['output_root'], settings['trial_directory'])
  pfacets.mkdir_p(output_path)

  log_path = (settings.get('log_path')
      or os.path.join(settings['output_root'], 'sparco.log'))
  logging.basicConfig(filename=log_path, filemode='a+', format=settings['log_format'],
      level=getattr(logging, settings['log_level'].upper()))
  sparco.trace.sp.Tracer.snapshot_interval = settings['snapshot_interval']

  sparco.trace.sparse_coder.Tracer.output_path = output_path
  if settings['coder']:
    traceutil.tracer.apply_tracer(sparco.trace.sparse_coder.Tracer,
        target=settings['coder'], **settings['SparseCoder'])
