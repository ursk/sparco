import logging
import os

import h5py

import traceutil.tracer

class Tracer(traceutil.tracer.Tracer):

  def __init__(self, **kwargs):
    kwargs.setdefault('snapshot_interval', 100)
    super(Tracer, self).__init__(**kwargs)

  def write_snapshot(self):
    self.write_basis()
    self.symlink_basis()
    # if self.create_plots:
    #   self.write_plots()

  def write_basis(self):
    self.basis_path = os.path.join(self.output_path, '{0}.h5'.format(self.target.t))
    h5 = h5py.File(self.basis_path, 'w')
    h5.create_dataset('phi', data=self.target.phi)
    h5.create_dataset('order', data=self.target.basis_sort_order)
    h5.create_dataset('variance', data=self.target.rootbufs.mean.a_variance)
    h5.create_dataset('l0', data=self.target.rootbufs.mean.a_l0_norm)
    h5.create_dataset('l1', data=self.target.rootbufs.mean.a_l1_norm)
    h5.create_dataset('l2', data=self.target.rootbufs.mean.a_l2_norm)
    h5.close()

  # create soft-link to basis file
  def symlink_basis(self):
    linkf = os.path.join(self.output_path, 'basis.h5')
    try:
      os.remove(linkf)
    except:
      pass
    os.symlink(self.basis_path, linkf)

  def write_plots(self):
    pass  # TODO

  def profile_targets(self):
    return ['learn_basis1', 'learn_basis2', 'load_patches', 'infer_coefficients']

  ########### CUSTOM DECORATORS

  # this stuff has to be implemneted here because I can't currently decorate
  # __init__ (since the tracer is applied AFTER object initialization)
  def t_run(tracer, orig, self, *args, **kwargs):
    tracer.dump_state(os.path.join(tracer.output_path, 'config.txt'))
    return orig(self, *args, **kwargs)

  def t_iteration(tracer, orig, self, *args, **kwargs):
    logging.info('Iteration #{0}'.format(self.t))
    ret = orig(self, *args, **kwargs)
    if (self.t > 0 and tracer.snapshot_interval
        and self.t % tracer.snapshot_interval == 0):
      tracer.write_snapshot()
    return ret

  wrappers = {
      'run': [t_run],
      'iteration': [t_iteration]
      }
