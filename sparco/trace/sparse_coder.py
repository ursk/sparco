import logging
import os
import traceutil.tracer
import sparco.trace.sp

# intended for RootSpikenet

class Tracer(traceutil.tracer.Tracer):

  def __init__(self, **kwargs):
    kwargs.setdefault('create_spikenet_directory', True)
    kwargs.setdefault('RootSpikenet_config', {})
    super(Tracer, self).__init__(**kwargs)

  ########### CUSTOM DECORATORS

  def t_run(tracer, orig, self, *args, **kwargs):
    logging.info('Beginning new SparseCoder run...')
    return orig(self, *args, **kwargs)

  def t_create_spikenet(tracer, orig, self, *args, **kwargs):
    config = args[0]
    tup = (self.t, config['num_iterations'],
      config['inference_settings']['lam'],
      config['inference_settings']['maxit'])
    logging.info('Round %d: num_iterations = %d, lam = %g, maxit = %d' % tup)
    if tracer.create_spikenet_directory:
      dir = "{0}_niter_{1}_lam_{2}_maxit_{3}".format(*tup)
      sn_output_path = os.path.join(tracer.output_path, dir)
    else:
      sn_output_path = tracer.output_path
    sn = orig(self, *args, **kwargs)
    traceutil.tracer.apply_tracer(sparco.trace.sp.Tracer,
        output_path=sn_output_path, target=sn, **tracer.RootSpikenet_config)
    return sn

  wrappers = {
      'run': [t_run],
      'create_spikenet': [t_create_spikenet],
      }
