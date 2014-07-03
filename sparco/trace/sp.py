import h5py

import traceutil.tracer

class Tracer(traceutil.tracer.Tracer):

  def write_snapshot(self):
    self.write_basis()
    self.symlink_basis()
    if self.create_plots:
      self.write_plots()

  def write_basis(self):
    self.basis_path = os.path.join(self.output_path, '{0}.h5'.format(self.t))
    h5 = h5py.File(self.basis_path, 'w')
    h5.create_dataset('phi', data=self.phi)
    h5.create_dataset('order', data=self.basis_sort_order)
    h5.create_dataset('variance', data=self.rootbufs.mean.a_variance)
    h5.create_dataset('l0', data=self.rootbufs.mean.a_l0_norm)
    h5.create_dataset('l1', data=self.rootbufs.mean.a_l1_norm)
    h5.create_dataset('l2', data=self.rootbufs.mean.a_l2_norm)
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

  def wrappers(self):

    def __init__(orig):
      def wrapped(self, *args, **kwargs):
        ret = orig(self, *args, **kwargs)
        self.snapshot_interval = Tracer.snapshot_interval
        self.dump_state(os.path.join(self.output_path, 'config.txt'))
        return ret
      return wrapped

    def iteration(orig):
      def wrapped(self, *args, **kwargs):
        ret = orig(self, *args, **kwargs)
        if self.t > 0 and self.snapshot_interval and self.t % self.snapshot_interval == 0:
          self.write_snapshot()
        return ret
      return wrapped

    return [__init__, iteration]
