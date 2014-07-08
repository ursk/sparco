from IPython import embed

import functools
import glob
import re
import os

import h5py
import pfacets
import numpy as np

class Sampler(object):

  def __init__(self, **kwargs):
    defaults = {
        'cache_size': 1000,
        'subsample': 1,
        'resample_cache': 1,
        'hdf5_data_path': ['data'],
        'time_dimension': 1,
        'patch_length': 128,
        'patch_filters': map(lambda f: functools.partial(f, self), Sampler.patch_filters),
        'channels': None
        }
    pfacets.set_attributes_from_dicts(self, defaults, kwargs)
    self.superpatch_length = self.patch_length * self.subsample * self.cache_size
    self.channel_dimension = int(not self.time_dimension)
    self.load_datasets()
    self.update_configuration_from_datasets()
    self.filter_datasets()
    self.patch_shape = (len(self.channels), self.patch_length)
    self.refresh_cache()

  # More initialization

  def get_data_mat(self, dataset):
    """Get the main matrix from an hdf5 file by descending `hdf5_data_path`"""
    return reduce(lambda a,k: a[k], self.hdf5_data_path, dataset)

  def load_datasets(self):
    """Open hdf5 streams to all h5 files in input directory."""
    if re.search('.h5', self.input_path):
      filenames = [self.input_path]
    else:
      filenames = glob.glob(os.path.join(self.input_path, '*.h5'))
    self.datasets = [ h5py.File(f, 'r') for f in filenames ]

  def update_configuration_from_datasets(self):
    """Pending decision about how to structure metadata"""
    if not self.channels:
      num_channels = self.get_data_mat(self.datasets[0]).shape[self.channel_dimension]
      self.channels = np.arange(num_channels)

  def filter_datasets(self):
    """Remove datasets that have a time length shorter than the configured minimum."""
    self.datasets = [ ds for ds in self.datasets
        if self.get_data_mat(ds).shape[self.time_dimension] >= self.superpatch_length]
    sizes = np.array([ ds['data'].shape[self.time_dimension] for ds in self.datasets ])
    self.relative_dataset_sizes = [ s / sizes.sum() for s in sizes ]

  # data sampling

  def get_random_data_matrix(self):
    """Return a dataset with probability weighted by its size relative to all data."""
    ds = self.datasets[pfacets.np.weighted_randint(self.relative_dataset_sizes)]
    return self.get_data_mat(ds)

  def get_patches(self, num):
    """Randomly select `num` valid patches from the cached superpatch.

    A valid patch is one that returns true for all functions in
    `self.patch_tests`.
    """
    if self.patches_retrieved > (self.cache_size * self.resample_cache):
      self.refresh_cache()
    self.patches_retrieved += num
    gen_func = functools.partial(pfacets.np.sample_array,
        self.superpatch, self.patch_length, axis=self.time_dimension)
    gen = iter(gen_func, None)
    # filt = functools.partial(self.patch_filter)
    return np.array(pfacets.generate_filtered(gen, self.patch_filter, num))

  def refresh_cache(self):
    """Cache a subset of all available data in memory.

    Randomly selects a dataset and a continuous segment of that dataset of
    length `superpatch_length`. Reads this into variable `superpatch`.
    """
    ds = self.get_random_data_matrix()
    start = np.random.randint(0, ds.shape[self.time_dimension] - self.superpatch_length)
    self.superpatch = ds[start:start+self.superpatch_length, self.channels]
    self.patches_retrieved = 0

  # TODO figure out what's going on with mean and variance here to fix this
  def patch_filter(self, patch):
    """Check if patch satisfies selection criteria.
    
    Selection criteria are Boolean-returning functions that take **kwargs as
    the only parameter. They receive `u_std`, `u_max_std`, and `u_max`.
    """ 
    return True  # TODO standin; accepts all patches
    u = (patch - mean) / std
    u_std = u.std(axis=1)
    max_u = np.max(np.abs(u))
    kwargs = {'u': u, 'u_std': u_std, 'max_u': max_u}
    tests = map(lambda f: f(patch), self.patch_filters)
    return not (False in tests)
        
  # patch filters

  def std_threshold(self, **kwargs):
    return np.any(u_std > self.max_std)

  def max_u_in_bound(self, **kwargs):
    return self.max_u_lower_bound < u_max < self.max_u_upper_bound

  patch_filters = [std_threshold, max_u_in_bound]
