"""
sparsify data stream with given basis and save to file
      
    
Process in overlapping chunks, tie chunks together at end to generate
one file. Generate spike times file from that file.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from time import time as now
import datetime
import ipdb
from mpi4py import MPI


home = os.path.expanduser('~')
path1 = os.path.join(home, 'Dropbox/nersc/csc/spikes/qn') # Cerberus
path2 = os.path.join(home, 'csc/spikes/qn')               # Hopper
if not path1 in sys.path: sys.path.append(path1)
if not path2 in sys.path: sys.path.append(path2)
print "pathes", path1, path2
import sparseqn
from sparseqn import sparseqn_batch
import sptools
import mp


# mpi info
rank = MPI.COMM_WORLD.Get_rank()
procs = MPI.COMM_WORLD.Get_size()
root = 0
mpi = (rank, procs, root)


class Logger(object):
    """
    Redirect stdout and stderr to a file and optionally echo to original stdout.
    """
    def __init__(self, stream, logFile, rank, echo=False):
        self.out = stream
        self.logFile = logFile
        self.echo = echo
        self.rank = rank
    
    def write(self, s):
        if len(s) == 1: return
        self.logFile.write('[%d] %s\n' % (self.rank, s))
        if self.echo:
            self.out.write('[%d] %s\n' % (self.rank, s))
        
    
    @staticmethod
    def start_logger(path, rank, echo, prefix):
        logFile = open(os.path.join(path, '%s.log' % prefix) , 'w+', 0)
        sys.stdout = Logger(sys.stdout, logFile, rank, echo)
        sys.stderr = Logger(sys.stderr, logFile, rank, echo)
    


def block_copy(inf, outf, in0=0, out0=0, size=None, bsize=10000):
    """
    Copy hdf5 dataset of size in bsize blocks
    
    inf      : input dataset
    outf     : output dataset
    in0      : input offset
    out0     : output offset
    size     : amount to copy
    bsize    : block size to use in copying
    """
    if size is None: size = len(inf)
    
    for i in range(0, size, bsize):
        j = min(i+bsize, size)
        # print 'Copying %d:%d to %d:%d' % (in0+i, in0+j, out0+i, out0+j)
        outf[out0+i:out0+j] = inf[in0+i:in0+j][:]
    


class ParallelSparsify(object):
    
    
    def __init__(self, recording, session,
                 basisf, reorder=True, subset=None,
                 method='mp', lam=1., positive=True, 
                 T=1000, s=0.005, dtype='high', postfix='',
                 subsample=1, channels=None, maxit=25,
                 maxdata=None, debug=True, debug_root=None):
        """
         recording  : eg. 'tiger/p6'
         session    : eg. 26
         basisf     : filename containing hdf5 dataset phi
         reorder    : whether to reorder basis from information in basisf
         subset     : use a subset of basis to sparsify
         method     : eg. 'mp', 'owlbfgs'
         lam        : sparsity penality wherever L1 regularization is used
         positive   : coefficients kept non-negative if True
         T          : block size of sparsification
         s          : sparsity for mp (.1 means 10% of coefficients are on)
         dtype      : eg. 'high', 'down', 'low'
         channels   : use this subset of channels
         maxdata    : for debugging, only sparsify this much of the data
         debug      : debug output
         debug_root : directory where debug plots are put
        """
        self.recording = recording
        self.session = session
        self.basisf = basisf
        self.reorder = reorder
        self.subset = subset
        self.method = method
        self.lam = lam
        self.positive = positive
        self.T = T
        self.s = s
        self.maxit = maxit
        self.dtype = dtype
        self.postfix = postfix
        self.subsample = subsample
        self.channels = channels
        self.maxdata = maxdata
        self.debug = debug
        self.debug_root = debug_root
        self.mu = None
        self.sigma = None
        
        #self.rootdir = os.path.join(os.path.expanduser('~'), 'sn', 'py')
        #self.debug_root = os.path.join(self.rootdir, 'daq', 'data/%s/%04d/out' % (recording, session))
        #self.dataroot = os.path.join(self.rootdir, 'daq', 'data/%s/%04d' % (recording, session))
        
        self.rootdir = os.path.join(os.path.expanduser('~'), 'Dropbox_outsource', 'nersc')
        self.dataroot = os.path.join(self.rootdir, 'data')
        
        self._load_data()
        self._load_basis()
    
    
    





    def _load_data(self):
        """
        Load electrophysiology data
        """
        if self.dtype == 'high':
            filen = 'all.%04d_micro.high-butter%s.h5' % (self.session, self.postfix)
        elif self.dtype == 'low':
            filen = 'all.%04d_micro_down.low-butter%s.h5' % (self.session, self.postfix)
        elif self.dtype == 'down':
            filen = 'all.%04d_micro_down%s.h5' % (self.session, self.postfix)
        elif self.dtype == 'climate':
            filen = 'gautam_testdata0.h5'
		elif self.dtype == 'ecog':
	            filen = 'xxxxxxxxxxxxx.h5'
        self.filen = os.path.join(self.dataroot, filen)
        
        print "opening file", self.filen
        self.inh5 = h5py.File(self.filen, 'r')
        self.data = self.inh5['data']
        if self.channels is None:
            self.channels = np.arange(self.data.shape[1])
        if self.dtype == 'climate':
            self.mean=np.array((0, ))
            self.var=np.array((1, ))
            self.std=np.array((1, ))
        else:
            self.mean = self.inh5['mean'][:][self.channels]
            self.var = self.inh5['var'][:][self.channels]
            self.std = np.sqrt(self.var - self.mean**2)
        self.D = len(self.data) / self.subsample
        
        
        if self.maxdata is not None:
            self.D = self.maxdata
        
        
        if rank == root:
            print '[%d] Dataset has %d channels and %d timepoints' % (rank, len(self.channels), self.D)
    
    
    def _load_basis(self, meanf=None, tighten=1):
        """
        Load basis and reorder. Optionally used a subset.
         'meanf', 'tighten' are parameters for the modified matching pursuit methods
        """
        print "trying to load basis", self.basisf
        basish5 = h5py.File(self.basisf, 'r')
        self.phi = basish5['phi'][:]
        
        
        self.C, self.N, self.P = self.phi.shape
        if self.C != len(self.channels):
            raise ValueError('Mismatch between data and basis')
        
        
        self.C, self.N, self.P = self.phi.shape        
        
        if self.reorder:
            try:
                order = basish5['order'][:]
                self.phi = self.phi[:,order].copy()                
            except:
                print 'No order attribute for basis. Not reordering'
        
        
        if self.subset is not None:
            self.N = len(self.subset)
            self.phi = self.phi[:,self.subset].copy()
        
        
        if meanf is not None:
            muh5 = h5py.File(meanf, 'r')
            self.mu = muh5['mu'][:]
            self.sigma = muh5['sigma'][:] / tighten
            muh5.close()
            
            if self.reorder:
                self.mu = self.mu[order].copy()
                self.sigma = self.sigma[order].copy()
            
            
            # remove bfs with mean zero
            nz = np.nonzero(self.mu)[0]
            self.N = len(nz)
            self.phi = self.phi[:,nz].copy()
            self.mu = self.mu[nz].copy()
            self.sigma = self.sigma[nz].copy()
            print 'Left with %d basis functions' % len(nz)
            
            
        
        basish5.close()
    
    
    def _create_output(self, filen, T):
        """
        Initialize coefficient output file. Use compression.
        """
        try:
            self.outh5 = h5py.File(filen, 'w')
        except:
            os.unlink(filen)
            self.outh5 = h5py.File(filen, 'w')            
        self.out = self.outh5.create_dataset('data', shape=(T, self.N),
                                             dtype=np.float32, compression='gzip',
                                             chunks=(min(T,1000), self.N)) 
            
    def open_output(self, filen):
        """
        Open coefficient output file (don't truncate).
        """
        self.outh5 = h5py.File(filen, 'r')
        self.out = self.outh5['data']
        
        
    
    
    
    
    def close(self):
        """
        Close open files
        """
        self.inh5.close()
        self.outh5.close()
        
    
    
    
    def get_data(self, t0, t1):
        """
        Return data as channels, time
        """
        t0 *= self.subsample
        t1 *= self.subsample
        return self.data[t0:t1:self.subsample,self.channels].astype(np.double).T
        
    
    
    
    def sparsify(self):
        """
        Split data into blocks, sparsify in parallel, and stitch
        together by performing an inference.
        """
        # split into blocks
        bsize = self.D / procs    
        bounds = np.zeros((procs, 2), dtype=np.uint32)
        b = 0
        for i in range(procs):
            bounds[i] = [b, b+bsize]
            b += bsize
        bounds[-1,1] = self.D
        if self.debug and rank == root:
            print ' block bounds: ', bounds
        
        
        # initialize output file
        fname = '%04d_micro_%s_coefficients%s' % (self.session, self.dtype, self.postfix)
        fileroot = os.path.join(self.dataroot, fname)
        filen = fileroot + '_proc%02d.h5' % rank
        
        
        t0 = bounds[rank,0]
        t1 = bounds[rank,1]
        pad = self.P - 1
        
        
        # create output file for block
        self._create_output(filen, T=t1 - t0 + pad)
        
        # run sparse coder
        self._run(t0=t0, t1=t1)
        
        
        # synchronize threads
        MPI.COMM_WORLD.Barrier()
        
        if rank == root:
            # stitch together        
            self._stitch(bounds, fileroot)
        
    
    
    
    def _run(self, t0, t1):
        """
        Sparsify data set
        t0, t1    : start and end times of block to sparsify
        
        
        
        [TODO] avoid re-reading pad of data
        [TODO] avoid transposing of data
        [TODO] avoid zero-ing out of xv
        """
        # if s is fractional and using mp, set to number of coefficients
        if self.s < 1.:
            self.s = int((self.P+self.T-1)*self.N * self.s)
        
        
        # allocate memory for sparsification
        # (sparsification routines use batches, so here we use 1 sample batches)
        x = np.zeros((1, self.C, self.T))
        xv = x[0]   # view of x
        a = np.zeros((1, self.T+self.P-1, self.N))
        av = a[0]   # view of a
        
        
        pad = self.P - 1
        
        
        # masking for mp uses inf to keep coefficient unchanged
        # whereas for the quasinewton methods, 0 indicates no change,
        # 1 indicate change
        if 'mp' in self.method:
            mask = np.zeros_like(av)
        else:
            mask = np.ones_like(av)
            mask_col = 0
        
        
        t = ot = 0
        D = t1 - t0
        finished = False
        
        frame = expired = 0
        
        
        if self.method == 'mp':
            pursuit = mp.ConvolutionalMatchingPursuit(self.phi, s=self.s, T=self.T,
                                                      positive=self.positive, debug=self.debug>1)
        if self.method == 'penalized-mp':
            print 'using penalized mp'
            extra = {'mu': self.mu, 'sigma': self.sigma, 'dt': 16}
            pursuit = mp.PenalizedMP(self.phi, s=self.s, T=self.T,
                                     positive=self.positive, debug=self.debug>1, extra=extra)
        
        
        if self.method == 'refractory-mp':
            print 'using refractory mp'
            extra = {'dt': 16}
            pursuit = mp.RefractoryMP(self.phi, s=self.s, T=self.T,
                                      positive=self.positive, debug=self.debug>1, extra=extra)
        
        
        tic = now()
        while not finished:
            if t+self.T > D:
                self.T = D - t   # [TODO] shouldn't change value of T
                finished = True
            
            
            xv[:,:self.T] = self.get_data(t0+t, t0+t+self.T)
            xv[:,self.T:] = 0        
            xv -= self.mean[:,None]
            xv /= self.std[:,None]
            
            
            if 'mp' in self.method:
                pursuit.run(x, A=a.transpose((0,2,1)), mask=mask.T)
            elif self.method == 'owlbfgs':
                sparsity_gain = 4
                A = sparseqn_batch(self.phi, x, Sin=a.transpose((0,2,1)),
                                   maxit=self.maxit, positive=self.positive,
                                   delta=0.00001, debug=self.debug>2,
                                   lam=sparsity_gain * self.lam,
                                   mask=mask.T)
                a[:] = A.transpose((0,2,1))
            else:
                raise ValueError('Bad method')
            
            
            if self.debug>2:
                self.debug_plot(xv, av, prefix='p%02d-%08d' % (rank, t))
            
            
            if not finished:
                if t == 0:
                    self.out[ot:ot+self.T] = av[:self.T]
            
            
                    # turn on masking for subsequent times
                    if 'mp' in self.method:
                        mask[:pad] = np.inf
                    else:
                        mask[:pad] = 0
                    mask_col = pad
                    ot += self.T
                else:
                    self.out[ot:ot+self.T-pad] = av[pad:self.T]
                    ot += self.T-pad
                
                
                if 'mp' in self.method:
                    av[:2*pad,:] = av[-2*pad:,:]
                    av[pad:].fill(0.)
                else:
                    av[:2*pad,:] = av[-2*pad:,:]
                    av[2*pad:].fill(0.)
                    
                t += self.T - pad                    
            else:
                self.out[ot:ot+self.T] = av[pad:pad+self.T]
                print '[%d] Completed with %d timepoints' % (rank, ot+self.T)
                if ot + self.T != D + self.P - 1:
                    print '[%d] Warning, length of coeff data != length of data' % rank
                    # resizing dataset will be slow
                    self.out.resize((ot+self.T, self.N))
                t += self.T                
            
            # print approximate time remaining
            frame += 1
            if frame % 10 == 0 and rank == root:
                expired = now() - tic
                left = str(datetime.timedelta(seconds=int((t1-t0) * expired / t - expired)))
                print '[%d] %d (left: %s)' % (rank, t, left)
            
        self.outh5.close()
        
        
    
    
    
    def _stitch(self, bounds, fileroot):
        """
        Stitch together final file
          bounds     : (number of blocks, start and end of blocks)
          fileroot   : for loading blocks and creating new file
        """
        # create new file
        fname = fileroot + '_merge.h5'
        try:
            sth5 = h5py.File(fname, 'w')
        except:
            os.unlink(fname)
            sth5 = h5py.File(fname, 'w')            
            
        sth5.create_dataset('phi', data=self.phi)
        
        
        
        sth5.create_dataset('mean', data=self.mean)
        sth5.create_dataset('var', data=self.var)
        sth5.create_dataset('std', data=self.std)                
        if self.mu is not None:
            sth5.create_dataset('mu', data=self.mu)
            sth5.create_dataset('sigma', data=self.sigma)            
        
        size = bounds[-1,1]+self.P-1
        out = sth5.create_dataset('data', shape=(size, self.N),
                                  dtype=np.float32, compression='gzip',
                                  chunks=(min(size,1000), self.N))
        # write out metadata
        out.attrs['length'] = self.D
        out.attrs['N'] = self.N
        out.attrs['C'] = self.C
        out.attrs['P'] = self.P
        out.attrs['T'] = self.T
        out.attrs['s'] = self.s
        out.attrs['bounds'] = bounds
        
        
        # open block files for reading
        files = [fileroot + '_proc%02d.h5' % i for i in range(len(bounds))]
        h5 = [h5py.File(f) for f in files]
        d = [h['data'] for h in h5]
        
        
        # allocate space
        pad = self.P-1
        
        x = np.zeros((1, self.C, 2*pad))
        xv = x[0]   # view of x
        a = np.zeros((1, 3*pad, self.N))
        av = a[0]   # view of a
        
        
        if 'mp' in self.method:
            mask = np.zeros_like(av)   
            mask[:pad] = np.inf
            mask[-pad:] = np.inf
            s = self.s * 2*pad / self.T
        else:
            mask = np.ones_like(av)    
            mask[:pad] = 0
            mask[-pad:] = 0
        
        
        # method
        if 'mp' in self.method:
            print 'Using %g as s for stitching' % s  
            if self.method != 'mp': raise NotImplementedError()
            pursuit = mp.ConvolutionalMatchingPursuit(self.phi, s=s, T=2*pad,
                                                      positive=self.positive, debug=self.debug>1)
        
        
        print 'Stitching: '
        # copy first block
        t = bounds[0,1] - bounds[0,0] 
        block_copy(d[0], out, size=t)
        print ' %d done' % t
        
        
        # stitch intermediate blocks
        for i in range(len(bounds)-1):
            # get non-overlap coefficients
            av.fill(0.)
            av[:pad] = d[i][-2*pad:-pad]
            av[-pad:] = d[i+1][pad:2*pad]
            
            tb = bounds[i,1]
            print 'Getting data: %d:%d' % (tb-pad, tb+pad)
            xv[:] = self.get_data(tb-pad, tb+pad)
            xv -= self.mean[:,None]
            xv /= self.std[:,None]
            
            
            A = a.transpose((0,2,1))
            if 'mp' in self.method:
                pursuit.run(x, A=A, mask=mask.T)
            else:
                A[:] = sparseqn_batch(self.phi, x, Sin=A, maxit=self.maxit, positive=self.positive,
                                      delta=0.00001, debug=self.debug, lam=self.lam, mask=mask.T)
                
            if self.debug:
                self.debug_plot(xv, av, prefix='stitch-%08d' % t)
            
            
            # write overlapping coefficients
            out[t:t+pad] = av[pad:2*pad]
            t += pad
            
            
            # write remainder of coefficients up to pad
            size = len(d[i+1]) - 2*pad
            block_copy(d[i+1], out, in0=pad, out0=t, size=size)
            t += size
            
            
            print ' %d done' % t
        
        
        # write remaining 
        out[t:t+pad] = d[-1][-pad:]
        t += pad
        
        
        print ' %d done' % t
        
        
        if t != len(out):
            print 'Warning: stitched dataset not correct length'
            print '  stitiched %d != out %d' % (t, len(out))
        
        
        print 'Finished stitching length %d dataset' % t
        
        
        for h in h5:
            h.close()
        
        
        sth5.close()
        
        
        # remove temporary files
        for f in files:
            try:
                os.unlink(f)
            except:
                print 'Failed to remove temporary file: ', f
        
    
    
    
    def reconstruct(self, A, volts=False):
        """
        Reconstruct data from sparse coefficients for debugging
         A      : coefficients  (time, basis)
         volts  : convert back to microvolts
        Returns:
         xhat   - reconstructed data (batch, channel, time)
        """
        T = len(A) - self.P + 1
        xhat = np.zeros((self.C, T))        
        for b in range(self.P):
            xhat += np.dot(self.phi[:,:,b], A[b:b+T].T)
        
        
        if volts:
            xhat *= self.std[:,None]
            xhat += self.mean[:,None]
        return xhat
    
    
    
    def generate_spike_file(self, filen, subset=None, threshold=.0):
        """
        Generate hdf5 file of spike times
        [TODO] I stopped using this so it hasn't been updated
        """
        spikeh5 = h5py.File(filen, 'w')
        d = spikeh5.create_group('data')
        
        
        d.create_dataset('phi', data=self.phi)
        d.attrs['length'] = self.D
        d.attrs['N'] = self.N
        d.attrs['C'] = self.C
        d.attrs['P'] = self.P
        d.attrs['T'] = self.T  # [TODO] this value will be wrong
        d.attrs['s'] = self.s
        d.attrs['threshold'] = threshold
        if self.mu is not None:
            spikeh5.create_dataset('mu', data=self.mu)
            spikeh5.create_dataset('sigma', data=self.sigma)            
        
        
        if subset is None:
            subset = range(self.N)
        d.attrs['subset'] = subset            
            
        for i in subset:
            print 'Generating spike dataset for basis %d' % i
            # load data
            # should take < .5gb for 1 hour recording
            # [TODO] slow because of compression and access method
            tic = now()
            v = self.out[:,i][:]
            print ' read took %g seconds' % (now() - tic)
            n = d.create_group('%d' % i)
            times = np.nonzero(v)[0]
            print ' %d times for basis %d' % (len(times), i)
            values = v[times]
            if threshold > 0.:
                #th = np.max(np.abs(values)) * threshold
                good = np.abs(values) > threshold
                times = times[good]
                values = values[good]
                if len(good) == 0:
                    times = values = np.array([])
                print ' reduced to %d times' % (len(times))
                ipdb.set_trace()
            try:
                n.create_dataset('t', data=times, dtype=np.uint32)
                n.create_dataset('a', data=values, dtype=np.float32)
            except:
                pass
                    
        
        
        spikeh5.close()
    
    
    
    def debug_plot(self, xv, av, prefix='dbg', figno=1):
        """
        Plot data, reconstruction
        """
        av = av.copy()
        if not self.debug: return
        self.gain = 1
        self.ascale = 1.
        xhatv = self.reconstruct(av)
        
        
        
        # plot data, reconstruction, coefficients
        plt.figure(figno)
        plt.clf()
        plt.ioff()
        sparsity = (av != 0).sum() / float(av.size)
        nrm = np.linalg.norm(xv)**2
        error = np.linalg.norm(xv - xhatv)**2
        snr = 10 * np.log10 ( nrm / error );
        plt.suptitle('sparsity = %g, error = %g, snr = %g' % (sparsity, error/nrm, snr))
        
        subplots = 4
        splt = 1
        plt.subplot(subplots,1,splt); splt += 1
        mx = np.max(np.abs(xv))
        plt.imshow(xv, vmin=-mx, vmax=mx, origin='lower', aspect='auto')
        plt.subplot(subplots,1,splt); splt += 1 
        plt.imshow(xhatv, origin='lower', vmin=-mx, vmax=mx, aspect='auto')
        plt.xticks([]); plt.yticks([])
        plt.subplot(subplots,1,splt); splt += 1
        plt.imshow(av.copy().T, aspect='auto')
        plt.subplot(subplots,1,splt); splt += 1
        pad = self.P-1
        av[pad] = 1
        av[2*pad] = 1
        av[-pad] = 1
        av[-2*pad] = 1                
        
        plt.spy(av.copy().T, aspect='auto')
        plt.xticks([]); plt.yticks([])        
        
        
        
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.92,
                            wspace=0.15, hspace=0.15)
        plt.draw()
        plt.ion()
        
        
        
        path = os.path.join(self.debug_root, 'movie')
        try: os.makedirs(path)
        except: pass
        plt.savefig(os.path.join(path, prefix + '.png'))
        plt.waitforbuttonpress(timeout=10)
        
    
    


def gautam_climate(recording, session, basisf, channels=None, method='owlbfgs'):
    p = {'recording': recording,
         'session': session,
         'basisf': basisf,
         'lam': 0.8,
         'maxit': 50,
         'positive': True,
         'method': method,
         'T': 1000,
         'dtype': 'climate',
         'postfix': 'test',
         'subsample': 2,
         'channels': channels,
         'maxdata': None,
         'debug': False}



    ps = ParallelSparsify(**p)
    ps.sparsify()




def ecog(recording, session, basisf, channels=None, method='owlbfgs'):
    p = {'recording': recording,
         'session': session,
         'basisf': basisf,
         'lam': 0.8,
         'maxit': 50,
         'positive': True,
         'method': method,
         'T': 1000,
         'dtype': 'ecog',
         'postfix': 'test',
         'subsample': 2,
         'channels': channels,
         'maxdata': None,
         'debug': False}



    ps = ParallelSparsify(**p)
    ps.sparsify()




if __name__ == '__main__':
    # start logger for MPI session
    
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-r', '--recording', dest='recording', help='recording session, eg. tiger/p6')            
    (options, args) = parser.parse_args()
    recording = options.recording
    
    home = os.path.expanduser('~')
    path = os.path.join(home, 'sn/py/daq/data')
    print 'Using path: ', path
    #Logger.start_logger(path, rank, echo=True, prefix='sparsify-%s' % recording.replace('/','-'))
    
    recording = 'foo'
    session = 66
    basisf = '/Users/urs/sn/py/spikes/out/gautamtest/basis.h5';
    basisf = '/Users/urs/sn/py/spikes/out/gautam_pca_whitened_negative/basis.h5'
    # data file name hardcoded in line 157, as dataroot + gautam_testdata0.h5, dataroot = '~', 'Dropbox', 'nersc', 'data' need to use Dropbox_outsource
    channels = range(64)
    gautam_climate(recording, session, basisf, channels)
    
# standard function call is 
# /Users/urs/anaconda/bin/mpirun -np 4 /Users/urs/anaconda/bin/python parallel_sparsify.py 
