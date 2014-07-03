from IPython import embed

import os
import pprint

# TODO necessary to use AGG?
import matplotlib
matplotlib.use('AGG', warn=False)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
import numpy as np
import h5py
from scipy.misc.pilutil import imsave
from scipy.signal import correlate

import sparco.sptools as sptools

class Writer(object):

  def __init__(self, **kwargs):
    os.makedirs(self.output_path)
    self.write_configuration(kwargs)

  def write_configuration(self, config):
    with open(os.path.join(self.output_path, 'config.txt'), 'w+') as f:
      pp = pprint.PrettyPrinter(indent=2, stream=f)
      pp.pprint(config)

  def write_snapshot(self):
    self.write_basis()
    self.symlink_basis()
    if self.create_plots:
      self.write_plots()

      # save basis to file (overwrite)

  # TODO clean up params, title
  def write_plots(self):
    # title='%05d l0:%0.3f e:%0.3f' % (self.t, np.mean(self.roota_l0_norm), error)
    title = 'Iteration {0}'.format(self.t)
    params = (self.mean_a_l0_norm, self.mean_a_l1_norm,
        self.mean_a_variance, self.mean_a_l2_norm)
    sorted_params = tuple( x[self.basis_sort_order] for x in params )
    sorted_phi = self.phi[:,self.basis_sort_order]
    sptools.grid_plot(sorted_phi, params=sorted_params,
        filename=os.path.join(self.output_path, '{0}.png'.format(self.t)))

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
