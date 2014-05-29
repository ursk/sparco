"""
A convolution based sparsenet of an array of time series

Key:
 phi - basis [channel, neuron, coefficients]
 A   - coefficients
 X   - patches of data

Note:
1. The kernels are convolution kernels. 
"""
import numpy as np
from mpi4py import MPI
from sptools import (vnorm, Logger, BasisWriter, attributesFromDict, set_paths, blur)
import learner
from time import time as now
import ipdb
import matplotlib.pyplot as plt


class Spikenet(object):
    
    def __init__(self, dims, db,
                 bs=10, niter=100, 
                 learner=None,
                 largs=None,
                 inference=None,
                 iargs={'lam': 0.1, 'maxit':50, 'debug':False},
                 disp=5, mpi=(0,1,0),
                 movie=False, prefix='vanilla',
                 logger=True, echo=True, plots=True):
        """
        Params
         dims (C, N, P, T)
          C - channels
          N - kernels
          P - time points in convolution kernel
          T - time points in raw data (T >> P)
         db - data source
        Inference and learning
         bs        - batch size to be divided among procs
         inference - inference method
         iargs     - inference optional args
         learner   - learning method
         largs     - learning optional args
        mpi       - (rank, procs, root)
        Output:
         disp      - save fig on this many iterations
         logger    - log output to file
         echo      - echo stdout to console
         movie     - use movie basis writer (for natural scenes)
         prefix    - prepend to output file names
         plots     - whether or not to output plots (due to memory leak in matplotlib)
        """
        attributesFromDict(locals())
        
        self.C, self.N, self.P, self.T = dims
        self.basis_dims = (self.C, self.N, self.P)
        self.coeff_dims = (self.N, self.T + self.P - 1)
        
        self.path = set_paths() # (URS) was hardcoded to ~/sn/py/spikes, change it?
        
        self.rank, self.procs, self.root = mpi
        if self.bs % self.procs != 0:
            raise ValueError('Batch size not multiple of number of procs')
            
        self.learner = learner(self.obj, dims, **largs)
        if 'lam' in iargs:
            self.lam = self.iargs['lam']
        else: self.lam = 0.
        
        # initialize logging and output
        if logger: Logger.start_logger(self.path['out'], self.rank, echo, self.prefix)
        self.basis_writer = BasisWriter(self.path, movie=self.movie,
                                        prefix=self.prefix, plots=plots)
        
    
    def init_basis(self, phi=None, file=None, filter=False, correlated=False):
        """
        Initialize basis to random unless phi or phi specified
         phi     - set basis to phi
         file    - load basis from file
         filter  - filter random normal
        """
        if self.rank == self.root:
            if file is not None:
                import h5py
                print 'Loading basis from  %s' % file
                h5 = h5py.File(file, 'r')
                phi = h5['phi'][:]
            elif phi is None:
                if correlated:  # vertical stripes + random pixel noise
                    phi = np.empty(self.basis_dims)
                    phi[:,:] = np.random.randn(self.basis_dims[1], self.basis_dims[2])[None,:]
                    phi += 0.5*np.random.randn(*self.basis_dims)
                else:
                    phi = np.random.randn(*self.basis_dims)
                if filter: phi = blur(self.phi)
            if phi.shape != self.basis_dims:
                raise ValueError('Warm start phi wrong dimensions')
            self.phi = phi            
            self.phi /= vnorm(self.phi)
        else:
            self.phi = np.empty(self.basis_dims)
    
    def code(self, t):
        """
        Generate string for output filenames
         t   - time step
        """
        code = ('%s-it=%06d,C=%d,N=%d,P=%d,T=%d,lam=%.2f,bs=%d,proc=%d') % (
            self.prefix, t, self.C, self.N, self.P, self.T,
            self.lam, self.bs, self.procs)
        return code
        
    
    def obj(self, phi, A, X, l1=False, recon=False):
        """
        Compute objective
         phi    - basis
         A      - coefficients  (batch, basis, time)
         X      - data (batch, channel, time)
         l1     - add L1 of A to objective if True
         recon  - return objective, derivative, reconstructed batches
        """
        dphi = np.zeros_like(phi)
        E = 0.
        xhat = np.zeros_like(X)        
        for pat in range(self.bs):
            for b in range(self.P):
                xhat[pat] += np.dot(phi[:,:,b], A[pat,:,b:b+self.T])
            
            dx = xhat[pat] - X[pat]
            E += 0.5 * np.linalg.norm(dx)**2
            
            for b in range(self.P):
                dphi[:,:,b] += np.dot(dx, A[pat,:,b:b+self.T].T)
                
        E /= self.bs
        dphi /= self.bs
        
        if l1: E += self.lam * abs(A).sum() / self.bs
        
        if recon:
            return E, dphi, xhat
        else:
            return E, dphi
        
    
    def learn(self):
        """
        Learn basis by alternative online minimization 
        """
        # allocate MPI buffers
        if self.rank == self.root:
            A = np.empty((self.bs,) + self.coeff_dims)
        else:
            A = np.array([])
        X = np.array([])
        parX = np.empty((self.bs / self.procs, self.C, self.T)) 
        
        sparse_tic = learn_tic = 0
        
        # broadcast basis to nodes
        MPI.COMM_WORLD.Bcast(self.phi, self.root)
        
        for self.t in range(self.niter):
            # get data batch and broadcast to nodes
            if self.rank == self.root:
                X = self.db.get_patches(self.bs)
            MPI.COMM_WORLD.Scatter(X, parX, self.root)
            
            # infer coefficients
            tic = now()
            parA = self.inference(self.phi, parX, **self.iargs)
            MPI.COMM_WORLD.Gather(parA, A, self.root)
            sparse_tic += now() - tic
            
            # update basis in parallel
            tic = now()
            self.learner.step(self.phi, A, X)
            learn_tic += now() - tic
            
            if self.rank == self.root:
                # dump step
                if self.t % self.disp == 0:
                    self.dump(A, X, self.t, sparse_tic, learn_tic)
                    sparse_tic = learn_tic = 0
            
    
    def dump(self, A, X, t, sparse_tic, learn_tic):
        """
        Dump iteration data to files
        """
        # l0 norm of all coefficients across batches
        l0norm = (A != 0.).sum().astype(np.float64) / np.prod(A.shape)
        
        # normalized reconstruction error
        E, gd, Xhat = self.obj(self.phi, A, X, recon=True, l1=False)
        z = self.obj(self.phi, np.zeros_like(A), X)[0]
        error = E / z
        snr = 10 * np.log10( 1 / error )
        
        out = '[%d] Inference: %.4fs, Learning: %.4fs, L0: %.8f, E: %f, SNR: %f'
        print out % (t, sparse_tic, learn_tic, l0norm, error, snr)
        
        # write basis to file
        self.basis_writer.write(self.phi, A, t, self.code(t),
                                error=error, X=X, Xhat=Xhat)
    
    
    

