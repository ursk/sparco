from IPython import embed
"""
Import various kinds of data

Some duplicate code between datasets
"""

import functools
import os
import numpy as np
import h5py
from scipy.ndimage.filters import gaussian_filter1d
import sptools
from time import time as now

class DB(object):

  def __init__(self, **kwargs):
    defaults = {
        'cache_size': 1000,
        'subsample': 1,
        'resample_cache': 1,
        'hdf5_data_path': ['data'],
        'time_dimension': 1,
        'patch_length': 128,
        'patch_filters': map(lambda f: functools.partial(f, self), DB.patch_filters),
        'channels': None
        }
    settings = sptools.merge(defaults, kwargs)
    for k,v in settings:
      setattr(self, k, v)
    self.superpatch_length= self.patch_length * self.subsample * self.cache_size
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
      self.channels = self.get_data_mat(self.datasets[0]).shape[self.channel_dimension]

  def filter_datasets(self):
    """Remove datasets that have a time length shorter than the configured minimum."""
    self.datasets = [ ds for ds in datasets
        if get_data_mat(ds)[self.time_dimension] >= self.superpatch_size]
    sizes = [ ds['data'].shape[self.time_dimension] for ds in self.datasets ]
    self.relative_dataset_sizes = [ s / sizes.sum() for s in sizes ]

  # data sampling

  def get_random_dataset(self):
    """Return a dataset with probability weighted by its size relative to all data."""
    return self.datasets[sptools.weighted_randint(self.relative_dataset_sizes)]

  def get_patches(self, num):
    """Randomly select `num` valid patches from the cached superpatch.

    A valid patch is one that returns true for all functions in
    `self.patch_tests`.
    """
    if self.patches_retrieved > (self.cache_size * self.resample_cache):
      self.refresh_cache()
    gen_func = functools.partial(sptools.sample_array,
        self.superpatch, self.patch_length, axis=1)
    gen = iter(gen_func, None)
    filt = functools.partial(self.patch_filter, self)
    return np.array(sptools.generate_filtered(gen, filt, num))

  def refresh_cache(self):
    """Cache a subset of all available data in memory.

    Randomly selects a dataset and a continuous segment of that dataset of
    length `superpatch_length`. Reads this into variable `superpatch`.
    """
    ds = self.get_random_dataset()
    start = np.random.randint(0, db.shape[1] - self.superpatch_length)
    self.superpatch = db[start:start+self.superpatch_length, self.channels]

  def patch_filter(self, patch):
    """Check if patch satisfies selection criteria.
    
    Selection criteria are Boolean-returning functions that take **kwargs as
    the only parameter. They receive `u_std`, `u_max_std`, and `u_max`.
    """ 
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
