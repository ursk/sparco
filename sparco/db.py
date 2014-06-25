from IPython import embed
"""
Import various kinds of data

Some duplicate code between datasets
"""
import os
import numpy as np
import h5py
# import scipy.misc
# import scipy.signal as sg
# from scipy.signal import correlate, lfilter
from scipy.ndimage.filters import gaussian_filter1d
from sptools import (attributesFromDict, weighted_randint)
# from matplotlib import pyplot as plt
# import matplotlib.mlab as mlab
from time import time as now

class DB(object):

    def __init__(self, dims=None, filenames=None, normalize='all',
                 cache=1000, resample=1, subsample=1,
                 channels=None, cull=0.0, artifact=0., std_threshold=0.,
                 maxcull=0., giveup=100, smooth=False, line=False, Fs=30000):
        """
        Settings:
        dims         : C, N, P, T - channel, neurons, convolution points, time
        filenames    : filename or list of hdf5 filenames
        normalize    : 'all' - make each channel mean zero and variance 1.
                       'patch' - normalize per patch, severe
                       'learn' - accumulate mean and variance info from patches
        subsample    : integer multiple to subsample data
        resample     : how many times to resample the cached data

        cull         : discard patches that have less than cull x the average variance
        artifact     : discard patches with voltages > +/- microvolts  (0. turns it off)
        maxcull      : discard patch if one of the channels is greater than maxcull std
        smooth       : smooth across lamina
        line         : remove line noise
        Fs           : sampling rate
        """
        attributesFromDict(locals())
        self.channels = self.channels or np.arange(self.C)
        self.C, self.N, self.P, self.T = dims
        self.C = len(self.channels)
        self.patch_shape = (len(self.channels), self.T)
        # if self.C != len(self.channels):
        #     raise ValueError()
        self.t = 0

        self._load_datasets()

        # some rules to discard patches
        self.select = False
        if (self.cull != 0. or self.artifact != 0. or
            self.maxcull != 0 or self.std_threshold != 0):
            self.select = True
        if self.artifact == 0: self.artifact = 1e16
        if self.maxcull == 0: self.maxcull = 1e16


    def _load_datasets(self):
        """
        Load datasets and collect any mean and variance information
        """
        # open datasets
        if type(self.filenames) == str:
            self.filenames = [self.filenames]
        self.numdb = len(self.filenames)
        self.db = []
        self.dset = []
        self.size = np.zeros(self.numdb, dtype=np.int)
        for i in range(self.numdb):
            #print 'Opening %s for reading.' % self.filenames[i]
            db = h5py.File(self.filenames[i], 'r')
            dset = db['data']
            size = len(dset)
            self.db.append(db)
            self.dset.append(dset)
            self.size[i] = size
            print 'Opened %s for reading.' % self.filenames[i]
            print 'Dataset has dimensions %s ' % (dset.shape,)

        self.weight = self.size / float(self.size.sum())

        # get means and variance of channels
        self.mean = np.zeros(self.C)
        self.var = np.ones(self.C)

        total = 0
        for i in range(self.numdb):
            if 'mean' in self.db[i].keys():
                total += self.size[i]
                self.mean += self.db[i]['mean'][:][self.channels] * self.size[i]
                self.var += self.db[i]['var'][:][self.channels] * self.size[i]
            else:
                print '  Dataset %s is missing mean info' % self.filenames[i]

        if total != 0:
            self.mean /= total
            self.var /= total
        else:
            print "zero total"
            if self.normalize is not None and self.normalize != 'patch':
                self.normalize = 'learn'
                self.samples = 0
                self.means = np.zeros(self.C)
                self.var = np.ones(self.C)
                print 'Dataset has no stored mean and variance of channels. Will accumulate this info.'

        self.std = np.sqrt(self.var - self.mean**2) # this produces NaNs. OOOOOPS!
        print "std", self.std


    def fetch(self):
        n = self.T * self.subsample * self.cache # batch size = 128 * 1 * cache[1000]

        # pick random dataset
        while True:
            db = weighted_randint(self.weight)
            if n < self.size[db]:
                r = np.random.randint(0, self.size[db]-n)
                break
            else:
                print 'Warning: requested batch size (%d) is larger than size of db (%d)' % (n, self.size[db])
                print 'Skipping this db...', db
                print self.size

        data = self.dset[db][r:r+n:self.subsample, self.channels][:].astype(np.float64)
        # optionally, remove line noise
        if self.line:
            data = remove_noise(data, Fs=self.Fs, subsample=self.subsample)

        # transpose to (channels, time)
        print "transpose to (channels, time) gives", data.T.shape #
        self.data = data.T

        # smooth data across lamina
        # if self.smooth:
        #     plt.figure(42)
        #     plt.subplot(2,1,1)
        #     plt.imshow(self.data[:,:250], origin='lower')
        #     plt.title('original')
        #     smoothed = gaussian_filter1d(self.data, self.smooth,
        #                                  axis=0, order=0, mode='reflect')
        #     plt.subplot(2,1,2)
        #     plt.imshow(smoothed[:,:250], origin='lower')
        #     plt.title('smoothed')
        #     plt.draw()
        #     self.data = smoothed
        #
        # if self.normalize == 'learn':
        #     L = self.data.shape[1]
        #     self.mean = (self.samples * self.mean +
        #                  L * np.mean(self.data, axis=1)) / (self.samples + L)
        #     self.var = (self.samples * self.var +
        #                  L * np.var(self.data, axis=1)) / (self.samples + L)
        #     self.std = np.sqrt(self.var - self.mean**2)
        #     self.samples += L
        #
        # if self.normalize == 'patch':
        #     # normalize data
        #     self.data -= self.mean[:,None]
        #     self.data /= self.std[:,None]

            # and normalize again
            # m = np.mean(self.data, axis=1)
            # v = np.std(self.data, axis=1)
            # self.data = (self.data - m[:,None]) / v[:,None]

        print 'Fetched %d samples from %s...' % (self.cache, self.filenames[db])



    def get_patches(self, npat):
        """
        Get npat patches
        """
        if self.t <= 0:
            self.fetch()
            self.t = self.cache * self.resample
            print "t zero", self.t

        maxT = self.data.shape[1] - self.T + 1
        if not self.select:
            # select patches uniformly
            r = np.random.randint(0, maxT, npat)
            r.sort()
            patches = np.array([self.data[:,i:i+self.T] for i in r])
        else:
            # pick only patches with certain criteria
            i = tries = 0
            patches = np.empty((npat, self.data.shape[0], self.T)) # empty is faster than zero but contains NaN

            while i < npat:
                # get a single random patch
                r = np.random.randint(0, maxT)
                patch = self.data[:,r:r+self.T] # (URS) self.data contains NaNs.
                # print "bye", patch #(URS) nans from the way mean and std are handled
                # is it above per time std threshold
                mxp = np.max(np.abs(patch))
                u = (patch - self.mean[:,np.newaxis]) / self.std[:,np.newaxis]
                #ipdb.set_trace()
                mxu = np.max(np.abs(u)) # (URS) mxu is a test criterion, has to be between cull and maxcull. Basically our data exceeds a lot of standard deviations and looks like an artifact.

                l = (mxu, self.cull, mxp, self.artifact)
                # is it above per channel std threshold
                passed_std = True
                if self.std_threshold > 0:
                    std = u.std(axis=1)
                    if np.any(std > self.std_threshold):
                        argmax = np.argmax(std)
                        s = (argmax+1, std[argmax], self.std_threshold)
                        print 'Channel %d: %g > %g std threshold' % s
                        passed_std = False

                
                # print "MXP: {0}".format(mxp)
                if (passed_std and mxp < self.artifact and
                    ((mxu > self.cull and mxu < self.maxcull) or
                    tries > self.giveup * npat)):
                    patches[i] = patch
                    i += 1
                    # print 'Accepted: %2.3f > %2.3f and %2.3f < %2.3f' % l
                else:
                    #print "Checks: passed_std", passed_std , "and mxp < self.artifact", mxp < self.artifact, "and mxu > self.cull",  mxu > self.cull, "and mxu < self.maxcull", mxu < self.maxcull
                    print "last test: mxu (max of u (patch-mean/std))", mxu, 'below maxcull', self.maxcull
                    #print "but self.mean ", self.mean, "and self.std ", self.std, 'were never computed!!'
                    print '[%d] Patch rejected' % i
                    if mxp > self.artifact:
                        print 'Artifact: %2.3f < %2.3f or %2.3f > %2.3f' % l

                tries += 1
                if tries > self.giveup * npat:
                    print 'Warning, culling and artifact criteria may be too stringent.'
                    print 'Getting new data...'
                    self.fetch()
                    self.t = self.cache * self.resample
                    tries = 0


        if self.normalize == 'all' or self.normalize == 'learn':
            patches -= self.mean[np.newaxis,:,np.newaxis]
            patches /= self.std[np.newaxis,:,np.newaxis]

        if self.normalize == 'patch':
             m = patches.mean(axis=2)
             std = patches.std(axis=2)
             patches -= m[:,:,None]
             patches /= std[:,:,None]

        if self.select:
            self.t -= tries
        else:
            self.t -= npat
        return patches
