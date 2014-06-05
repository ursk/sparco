"""
different learning methods

1. Jack's angle chasing method
2. Coordinate-wise descent
3. Parallel block coordinate-wise descent
4. Truncated projected gradient
"""
import numpy as np
import matplotlib.pyplot as plt
from sptools import attributesFromDict, debug, norm, vnorm
from time import time as now
import mpi4py.rc
mpi4py.rc.profile('MPE')
from mpi4py import MPI, MPE
import scipy.sparse as sparse
from scipy.signal import lfilter
# import ipdb
import h5py
import os

class Learner(object):
    pass

def angles(phi0, phi):
    """
    Returns angles between basis vectors phi0 and phi in
    degrees. Assumes norm one.
    """
    dots = np.array([np.sum(phi[:,i]*phi0[:,i])
                     for i in range(phi.shape[1])])
    angle = np.arccos(dots) * 180. / np.pi
    angle[np.isnan(angle)] = 0.
    return angle


class AngleChasing(Learner):
    """
    Jack's method, try to keep basis phi (treated as single vector) changing
    a specified number of degrees on each step.
    """

    def __init__(self, obj, dims, eta=.0001, up=1.01, down=.99, target=5.,
                 c=100, cmax=1, smooth=False, thresh=10, mpi=(0,1,0)):
        """
        obj            - returns E, dE/dphi
        dims           - C, N, P, T
        eta            - update rate
        up             - eta increase
        down           - eta decrease
        target         - target angle in degress of max angle change
        thresh         - discard spurious updates to basis
        c              - center basis functions every few steps
        cmax           - max shift in basis function on centering step
        mpi            - rank, procs, root
        """
        attributesFromDict(locals())
        self.C, self.N, self.P, self.T = dims
        self.rank, self.procs, self.root = mpi
        if self.rank != self.root: return
        
        self.debug = True
        self.t = 0
        
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!this is learner with thresh", self.thresh

    def update(self, phi, a, X):
        """
        Inputs
         phi   - current basis   (modified on return)
         a     - coefficients
         X     - data
        """
        oldphi = phi.copy()
        obj, dphi = self.obj(phi, a, X)
        
        newphi = phi - self.eta * dphi
        newphi /= vnorm(newphi)
        
        dot = np.sum(newphi[:]*oldphi[:])/ (np.linalg.norm(newphi) * np.linalg.norm(oldphi))
        angle = np.arccos(dot) * 180. / np.pi
        if np.isnan(angle): angle = 0.
        #angle = max(angles(phi, oldphi))
        if angle < self.thresh:
            phi[:] = newphi[:]
        else:
            print 'Update to phi too large. Rejecting.'
        if angle < self.target:
            self.eta *= self.up
        else:
            self.eta *= self.down

        debug('[%d] eta = %g, angle = %g' % (self.t, self.eta, angle))
        self.t += 1

    def center(self, phi):
        """
        Shift each basis function to its center of mass by a maximum
        amount of cmax.

        Optionally, smooth basis functions.

        Center of mass is defined using sum of squares.
        """
        for n in range(self.N):
            s = np.sum(phi[:,n]**2, axis=0)
            total = np.sum(s)
            if total == 0.: continue
            m = int(np.round(np.sum(np.arange(self.P) * s)/total))
            shift = self.P/2 - m
            shift = np.sign(shift) * min(abs(shift), self.cmax)
            phi[:,n] = np.roll(phi[:,n], shift, axis=1)
            if shift > 0:
                phi[:,n,0:shift] = 0.
            elif shift < 0:
                phi[:,n,shift:] = 0.
            else:
                continue
            phi[:,n] /= np.linalg.norm(phi[:,n])
            print 'Shifting %d by %d' % (n, shift)

        if self.smooth:
            a = 1
            b = [0.25, .5, 0.25]
            for n in range(self.N):
                phi[:,n] = lfilter(b, a, phi[:,n], axis=1)

    def step(self, phi, a, X):
        
        if self.rank == self.root:
            self.update(phi, a, X)
            if self.c is not None and self.t % self.c == 0:
                self.center(phi)
        MPI.COMM_WORLD.Bcast(phi, self.root)

class GroupMP(AngleChasing):

    def __init__(self, obj, dims, eta=.0001, up=1.01, down=.99, target=5.,
                 c=100, cmax=1, mpi=(0,1,0), extra=None):
        super(GroupMP, self).__init__(obj, dims, eta, up, down, target, c, cmax, mpi)
        self.extra = extra
        self.gsize = extra['gsize']
        self.orthonormal = extra['orthonormal']
        self.groups = self.N / self.gsize

    def update(self, phi, a, X):
        """
        Inputs
         phi   - current basis   (modified on return)
         a     - coefficients
         X     - data
        """
        super(GroupMP, self).update(phi, a, X)

        if not self.orthonormal: return
        
        # orthonormalize groups with Gramm-Schmidt
        for i in range(0, self.N, self.gsize):
            for j in range(i+1, i+self.gsize):
                for k in range(i,j):
                    phi[:,j] -= (phi[:,j] * phi[:,k]).sum() * phi[:,k]
                phi[:,j] /= np.linalg.norm(phi[:,j])

        if False:
            for g in range(self.groups):
                print 'Testing orthonormality for group %d' % g
                s = g*self.gsize
                for i in range(s,s+self.gsize):
                    for j in range(s, s+self.gsize):
                        if i >= j: continue
                        print '[%d,%d] %g' % (i,j, (phi[:,i]*phi[:,j]).sum())


class PenalizedMP(AngleChasing):

    def __init__(self, obj, dims, eta=.0001, up=1.01, down=.99, target=5.,
                 c=100, cmax=1, mpi=(0,1,0), extra=None):
        super(PenalizedMP, self).__init__(obj, dims, eta, up, down, target, c, cmax, mpi)
        self.extra = extra
        self.mu = extra['mu']
        self.sigma = extra['sigma']
        self.dt = extra['dt']
        self.switch = extra['switch']
        self.mupath = extra['mupath']        
        self.penalized = False

    def update(self, phi, a, X):
        """
        Inputs
         phi   - current basis   (modified on return)
         a     - coefficients
         X     - data
        """
        # switch between penalized and unpenalized learning
        if self.t % self.switch == 0:
            self.penalized = not self.penalized
            # self.penalized = True
            self.extra['penalized'] = self.penalized

        # update basis function 
        oldphi = phi.copy()
        obj, dphi = self.obj(phi, a, X)
        
        phi -= self.eta * dphi
        phi /= vnorm(phi)

        dot = np.sum(phi[:]*oldphi[:])/ (np.linalg.norm(phi) * np.linalg.norm(oldphi))
        angle = np.arccos(dot) * 180. / np.pi
        if np.isnan(angle): angle = 0.
        if angle < self.target:
            self.eta *= self.up
        else:
            self.eta *= self.down

        debug('[%d] eta = %g, angle = %g' % (self.t, self.eta, angle))

        # don't update means if unless penalized learning step
        if not self.penalized:
            self.t += 1
            return
        
        # find mean average change from means for non-zero coefficients

        dmu = (a.max(axis=2).max(axis=0) - self.mu)
        #dmu = ((a>0) * (a - self.mu[None,:,None])).sum(axis=0).sum(axis=1) / (a>0).sum()
        self.mu += .1 * dmu
        self.mu[self.mu < 0] = 0
        h5 = h5py.File(os.path.join(self.mupath, 'mu.h5'), 'w')
        h5.create_dataset('mu', data=self.mu)
        h5.create_dataset('sigma', data=self.sigma)
        h5.close()
        debug('[%d] mu = %s' % (self.t, str(self.mu)))


        if self.t % 10 == 0:
            plt.figure(25)
            plt.clf()
            plt.plot(self.mu)
            plt.title('Means')
            plt.draw()
            plt.ion()

        if self.t % self.switch == -1:
            print 'Creating coeff histogrms'
            N = phi.shape[1]
            rows = cols = int(np.ceil(np.sqrt(N)))
            if (rows-1) * cols > N: rows -= 1
            fig = plt.figure(40)
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
                                top=0.95, wspace=0.30, hspace=0.30)    
            plt.clf()
            k = 1
            plt.ioff()
            for i in range(N):
                coeff = a[:,i,:].flatten()
                coeff = coeff[coeff>0]
                if len(coeff) == 0: continue
                plt.subplot(rows, cols, k); k += 1
                n, bins, patches = plt.hist(coeff, 20, normed=True, facecolor='green',
                                            alpha=0.75)
                plt.title(i)
            plt.suptitle(self.t)
            plt.draw()
            plt.ion()
        
        self.t += 1


        


class CoordinatewiseDescent(Learner):
    """
    Block coordinatewise descent.
    See Bertsekas (2000) Proposition 2.7.1, Mairal et al. (2010).
    Also Tseng (2001)

    [TODO] Algorithm is dominated by update of A
      - Change ordering of a so slicing is done on first index
      - Use fact that a is sparse
      - Use CUDA to do update
    [TODO] Rank Q updates to matrix inverses (Sherman-Morrison)
    [TODO] Add epsilon to diagonal of A periodically to condition the
    Hessian as necessary
    """

    def __init__(self, obj, dims, maxiter=10, tol=1e-4, epsilon=1.,
                 t0=10, rho=15, mpi=(0,1,0)):
        """
        obj            - returns E, dE/dphi (not used)
        dims           - C, N, P, T
        maxiter        - maximum coordinate sweeps
        tol            - tolerance in percentage change of objective
        epsilon        - initial diagonals of A
        t0             - annealing term
        rho            - annealing term, bigger more gradual start
        memory         - use epoch memory
        mpi            - rank, procs, root
        """
        attributesFromDict(locals())
        self.C, self.N, self.P, self.T = dims
        self.rank, self.procs, self.root = mpi
        if t0 < 2: raise ValueError('t0 < 2')
        if self.rank != self.root: return

        bdim = (self.C, self.N, self.P)

        # initialize A and B
        self.A = np.zeros((self.N, self.P, self.N, self.P))
        self.Av = self.A.reshape((self.N, self.P, self.N * self.P))
        self.tmpA = np.empty_like(self.A)        
        for i,j in np.ndindex((self.N, self.P)):
            self.A[i,j,i,j] = self.epsilon
        self.U = np.empty((self.N, self.P, self.P))
        self.dphi = np.empty(bdim)
        self.dphiv = self.dphi.reshape((self.C, self.N * self.P))
        self.B = np.empty(bdim)
        self.tmpB = np.empty(bdim)
        self.norm = 0.

        self.oldphin = np.empty((self.C, self.P))

        self.t = 0
        self.debug = True

    def debug_timing(self, a, X):
        """
        Some attempts to speed up stuff
        """
        # a in Fortran mode to speed up slicing
        tic = now()
        af = a.copy('F')
        self.tmpB.fill(0.)
        for b in range(self.P):
            self.tmpB[:,:,b] += np.tensordot(X, af[:,:,b:b+self.T], ([0,2],[0,2]))
        self.tmpB /= Q    
        toc = now() - tic
        print 'B update fortran: ', toc

        self.tmpA.fill(0.)
        tic = now()
        for n,p in np.ndindex(self.P, self.P):
            if n <= p:
                self.tmpA[:,n,:,p] += np.tensordot(af[:,:,n:n+self.T],
                                                   af[:,:,p:p+self.T], ([0,2], [0,2]))
            else:
                self.tmpA[:,n,:,p] = self.tmpA[:,p,:,n].T 
        self.tmpA /= Q
        toc = now() - tic
        print 'A update fortran: ', toc

        # use sparse column format for a's
        # unfortunately, there is no tensor dot equivalent
        self.tmpA.fill(0.)
        tic = now()
        asp = [sparse.csc_matrix(m) for m in a]
        for q in range(Q):
            for n,p in np.ndindex(self.P, self.P):
                if n <= p:
                    self.tmpA[:,n,:,p] += np.dot(a[q][:,n:n+self.T],
                                                 a[q][:,p:p+self.T].T)
                else:
                    self.tmpA[:,n,:,p] = self.tmpA[:,p,:,n].T 
        self.tmpA /= Q
        toc = now() - tic
        print 'A update sparse: ', toc

        # transpose for better memory access, similar to Fortran
        self.tmpA.fill(0.)
        aT = a.transpose((2,0,1)).copy()
        tic = now()
        for n,p in np.ndindex(self.P, self.P):
            if n <= p:
                self.tmpA[:,n,:,p] += np.tensordot(aT[n:n+self.T],
                                                   aT[p:p+self.T], ([0,1], [0,1]))
            else:
                self.tmpA[:,n,:,p] = self.tmpA[:,p,:,n].T 
        self.tmpA /= Q
        toc = now() - tic
        print 'A update transpose: ', toc
        
        
    def objective(self, phi, dphi):
        """
        compute objective and derivative
        """
        c = np.tensordot(phi, self.A, 2)
        dphi[:] = -self.B + c
        obj = np.sum(phi*(-self.B + 0.5 * c))
        return obj

    def update(self, phi, a, X):
        """
        Inputs
         phi   - current basis   (modified on return)
         a     - coefficients
         X     - data
        """
        # initialize B?
        if self.t == 0: self.B[:] = self.epsilon * phi

        # update A and B
        Q = len(X)
        decay = (1 - 1./max(1, (self.t + self.t0) ))**self.rho
        self.norm = decay * self.norm + 0.5/Q * np.linalg.norm(X)**2

        self.tmpB.fill(0.)
        for b in range(self.P):
            self.tmpB[:,:,b] += np.tensordot(X, a[:,:,b:b+self.T], ([0,2],[0,2]))
        self.tmpB /= Q
        self.B *= decay
        self.B += self.tmpB

        # [TODO] this is very slow
        tic = now()
        self.tmpA.fill(0.)
        for n,p in np.ndindex(self.P, self.P):
            if n <= p:
                self.tmpA[:,n,:,p] += np.tensordot(a[:,:,n:n+self.T],
                                                   a[:,:,p:p+self.T], ([0,2], [0,2]))
            else:
                self.tmpA[:,n,:,p] = self.tmpA[:,p,:,n].T 
        self.tmpA /= Q
        self.A *= decay
        self.A += self.tmpA
        toc = now() - tic
        print '   A update: %gs' % toc

        # compute inverse of A blocks
        tic = now()
        for n in range(self.N):
            self.U[n] = np.linalg.inv(self.A[n,:,n,:])
        toc = now() - tic
        print '   Inverses: %gs' % toc
            
        # initialize objective and derivative
        obj = oldobj = self.objective(phi, self.dphi)

        tic = now()
        for iter in range(self.maxiter):
            # cycle through local basis
            for n in range(self.N):
                self.oldphin[:] = phi[:,n]
                phi[:,n] -= np.dot(self.dphi[:,n], self.U[n])
                norm = np.linalg.norm(phi[:,n])
                if norm > 1:
                    phi[:,n] /= norm
                elif norm == 0.:
                    phi[:,n] = np.random.randn(*phi[:,n].shape)
                    debug('   Reinitializing phi[%d] in step %d' % (n, iter))
                # [TODO] Slow!
                self.dphi += np.tensordot(phi[:,n] - self.oldphin, self.A[n], ([1],[0]))
            obj = self.objective(phi, self.dphi)
            stop = int(abs(obj/oldobj - 1) < self.tol)
            oldobj = obj                                
            if stop: break
        toc = now() - tic
        print '   Coordinate sweeps: %d in %gs' % (iter, toc)
        if self.debug:
            x = (self.t, obj, obj + self.norm, iter, decay)
            debug('[%d] obj-norm = %f, obj = %f, iter = %d, decay = %g' % x)
        self.t += 1

    def step(self, phi, a, X):
        if self.rank == self.root:
            self.update(phi, a, X)
        MPI.COMM_WORLD.Bcast(phi, self.root)

             

class ParallelCoordinatewiseDescent(Learner):
    """
    Block coordinatewise descent.
    See Bertsekas (2000) p230, Mairal et al. (2010)

    To simplify MPI, phi is transposed to basis, channel, time.

    See comments to CoordinatewiseDescent

    [TODO] Update for correct basis function updating using inverse's and line search
    [TODO] Epoch feature doesn't work
    [BUG] When using many processors, starting steps are very finicky
    """

    def __init__(self, obj, dims, maxiter=10, tol=1e-4, epsilon=1.,
                 t0=25, rho=15, memory=False, et=100, mpi=(0,1,0), mpe=False):
        """
        obj            - returns E, dE/dphi (not used)
        dims           - C, N, P, T
        maxiter        - maximum coordinate sweeps
        tol            - tolerance in percentage change of objective
        epsilon        - initial diagonals of A
        t0             - annealing term
        rho            - annealing term, bigger more gradual start
        memory         - use epoch memory
        et             - epoch times
        mpi            - rank, procs, root
        """
        attributesFromDict(locals())
        self.C, self.N, self.P, self.T = dims
        self.rank, self.procs, self.root = mpi
        if self.N % self.procs != 0:
            raise NotImplementedError('Basis functions not multiple of procs')
        if t0 < 2: raise ValueError('t0 < 2')
        self.Nl = self.N / self.procs

        # initialize A and B
        if self.rank == self.root:
            self.A = np.zeros((self.N, self.P, self.N, self.P))
            for i,j in np.ndindex((self.N, self.P)):
                self.A[i,j,i,j] = self.epsilon
            self.U = np.empty((self.N, self.P, self.P))                            
            self.phi = np.empty((self.N, self.C, self.P))
            self.phiT = self.phi.transpose((1,0,2))  # a view of phi
            self.dphi = np.empty_like(self.phi)                                    
            self.B = np.empty_like(self.phi)                            
            self.Ac = np.empty((self.procs, self.Nl, self.P, self.Nl, self.P))
            self.norm = 0.

            self.tmpA = np.zeros_like(self.A)
            self.tmpB = np.zeros_like(self.B)

            if self.memory:
                self.epochA = np.zeros_like(self.A)
                self.epochB = np.zeros_like(self.phi)
                self.epochnorm = 0.
                self.tmpnorm = 0.
                self.epoch = False
        else:
            self.Ac = np.array([])
            self.phi = np.array([])
            self.dphi = np.array([])

        # scatter buffers
        self.phil = np.empty((self.Nl, self.C, self.P))
        self.dphil = np.empty((self.Nl, self.C, self.P))
        self.Al = np.empty((self.Nl, self.P, self.Nl, self.P))

        self.scale = np.empty((self.Nl, 1, self.P))
        self.oldphin = np.empty((self.C, self.P))
        self.stop = np.empty((1,), dtype=np.int32)
        
        self.t = 0
        self.debug = True

        # logging
        MPE.initLog(logfile='step')
        
        self.step_begin = MPE.newLogEvent("Step-Begin", "yellow")
        self.step_end = MPE.newLogEvent("Step-End", "pink")
        self.init = MPE.newLogState("Initialization", "blue")
        self.scatter = MPE.newLogState("Scatter", "orange")
        self.gather = MPE.newLogState("Gather", "red")                
        self.broadcast = MPE.newLogState("Broadcast", "green")
        self.comp = MPE.newLogState("Computation", "cyan")        
        if not mpe:
            MPI.Pcontrol(0)

    def objective(self):
        """
        compute objective and derivative
        """
        c = np.tensordot(self.phiT, self.A, 2).transpose((1,0,2))
        self.dphi[:] = -self.B + c
        obj = np.sum(self.phi*(-self.B + 0.5 * c))            
        return obj

    def step(self, phi0, a, X):
        """
        Inputs
         phi   - current basis   (modified on return)
         a     - coefficients
         X     - data
        """
        with self.step_begin: pass

        with self.init:
            if self.rank == self.root:
                # initialize phi and possibly B
                self.phiT[:] = phi0
                if self.t == 0: self.B[:] = self.epsilon * self.phi

                # check epoch
                if self.memory and self.t % self.et == 0:
                    if not self.epoch:
                        debug('Reached epoch, resetting.')
                        self.A -= self.epochA
                        self.B -= self.epochB
                        self.norm -= self.epochnorm
                        self.epochA.fill(0.)
                        self.epochB.fill(0.)
                        self.epochnorm = 0.
                    else:
                        debug('Stopped collection')
                    self.epoch = not self.epoch

                # update A and B
                Q = len(X)
                decay = (1 - 1./max(1, (self.t + self.t0) ))**self.rho

                self.tmpnorm = 0.5/Q * np.linalg.norm(X)**2
                self.norm = decay * self.norm + self.tmpnorm

                self.tmpB.fill(0.)
                for b in range(self.P):
                    self.tmpB[:,:,b] += np.tensordot(a[:,:,b:b+self.T], X, ([0,2],[0,2]))
                self.tmpB /= Q    
                self.B = decay * self.B + self.tmpB

                self.tmpA.fill(0.)
                for n,p in np.ndindex(self.P, self.P):
                    self.tmpA[:,n,:,p] += np.tensordot(a[:,:,n:n+self.T],
                                                       a[:,:,p:p+self.T], ([0,2], [0,2]))
                self.tmpA /= Q
                self.A = decay * self.A + self.tmpA

                for n in range(self.N):
                    self.U[n] = np.linalg.inv(self.A[n,:,n,:])

                # accumulate epoch
                if self.memory and self.epoch:
                    self.epochA = decay * self.epochA + self.tmpA
                    self.epochB = decay * self.epochB + self.tmpB
                    self.epochnorm = decay * self.epochnorm + self.tmpnorm
                    debug('Collecting')

                # prepare diagonal blocks of A for scatter
                j = 0
                for n in range(0, self.N, self.Nl):
                    self.Ac[j] = self.A[n:n+self.Nl,:,n:n+self.Nl,:]
                    j += 1

                # initialize objective and derivative
                obj = oldobj = self.objective()

        # scatter A and dphi
        with self.scatter:
            MPI.COMM_WORLD.Scatter(self.Ac, self.Al, self.root)
            MPI.COMM_WORLD.Scatter(self.dphi, self.dphil, self.root)

        # initialize local basis
        self.phil[:] = phi0[:,self.rank*self.Nl:(self.rank+1)*self.Nl].transpose((1,0,2))

        for n in range(self.Nl):
            self.scale[n,:] = 1./np.diag(self.Al[n,:,n,:])[np.newaxis,:]

        for iter in range(self.maxiter):
            # cycle through local basis
            with self.comp:            
                for n in range(self.Nl):
                    self.oldphin[:] = self.phil[n]
                    # [TODO] modify to use U[n]
                    self.phil[n] -= self.scale[n] * self.dphil[n]
                    norm = np.linalg.norm(self.phil[n])
                    if norm > 1:
                        self.phil[n] /= norm
                    elif norm == 0.:
                        self.phil[n] = np.random.randn(*self.phil[n].shape)
                        debug('  Reinitializing phi[%d] in step %d' % (n, iter))
                    self.dphil += np.tensordot(self.phil[n] - self.oldphin,
                                               self.Al[n], ([1],[0])).transpose((1,0,2))

            # [TODO] Need to do a line search here
            # gather phi, compute objective and derivatives, stop?
            with self.gather:
                MPI.COMM_WORLD.Gather(self.phil, self.phi, self.root)

            with self.comp:
                if self.rank == self.root:
                    obj = self.objective()
                    self.stop[0] = int(abs(obj/oldobj - 1) < self.tol)
                    oldobj = obj
            with self.broadcast:
                MPI.COMM_WORLD.Bcast(self.stop, self.root)
            if self.stop[0]: break

            # scatter derivatives to nodes
            with self.scatter:
                MPI.COMM_WORLD.Scatter(self.dphi, self.dphil, self.root)

        # broadcast all of phi to nodes before returning
        if self.rank == self.root:
            if self.debug:
                angle = angles(phi0, self.phiT)
                x = (self.t, obj, obj + self.norm, iter, decay, np.max(angle),
                     np.min(angle), np.mean(angle), np.std(angle))
                debug(('[%d] obj-norm = %f, obj = %f, iter = %d, decay = %g, ' +
                       'dphi max=%f, min=%f, mean=%f, std=%f degrees') % x)
            phi0[:] = self.phiT
        with self.broadcast:
            MPI.COMM_WORLD.Bcast(phi0, self.root)
        self.t += 1

        with self.step_end: pass


class ProjectedGradientDescent(Learner):
    """
    [TODO] Make code more efficient
    """
    
    def __init__(self, obj, dims, maxiter=20, tol=1e-8,
                 epsilon=0.001, t0=2, rho=15,
                 s=1, beta=.5, sigma=1e-4, mpi=(0,1,0)):
        """
        obj            - returns E, dE/dphi
        dims           - C, N, P, T
        s, beta, sigma - Armijo params (see Bertsekas)
        maxiter        - maximum number of steps in line search
        """
        attributesFromDict(locals())
        self.C, self.N, self.P, self.T = dims
        self.rank, self.procs, self.root = mpi

        if self.rank != self.root: return
        
        # set A to small multiple of identity
        self.A = np.zeros((self.N, self.P, self.N, self.P))
        self.tmpA = np.empty_like(self.A)        
        for i,j in np.ndindex((self.N, self.P)):
            self.A[i,j,i,j] = self.epsilon
            
        self.B = np.zeros(dims[:-1])
        
        self.t = 0
        
        self.maxarmijo = 10
        self.otol = 1e-8
        self.debug = True
        self.trace = True

        self.k0 = 0
        self.backstep = 2

    def objective(self, phi):
        """
        compute objective and derivative
        """
        c = np.tensordot(phi, self.A, 2)
        dphi = -self.B + c
        obj = np.sum(phi*(-self.B + 0.5 * c))
        
        return obj, dphi

    def update(self, phi0, a, X):
        """
        Inputs
         phi0   - current basis
         a      - coefficients
         X      - data

        See Bertsekas (2000) p230
        """
        Q = len(X)        
        decay = (1 - 1./max(1, (self.t + self.t0) ))**self.rho
        self.B *= Q * decay
        self.A *= Q * decay

        # update data dependent term
        for b in range(self.P):
            self.B[:,:,b] += np.tensordot(X, a[:,:,b:b+self.T], ([0,2],[0,2]))
        self.B /= Q

        # update outer product term
        tic = now()
        self.tmpA.fill(0.)
        for n,p in np.ndindex(self.P, self.P):
            if n <= p:
                self.tmpA[:,n,:,p] += np.tensordot(a[:,:,n:n+self.T],
                                                   a[:,:,p:p+self.T], ([0,2], [0,2]))
            else:
                self.tmpA[:,n,:,p] = self.tmpA[:,p,:,n].T 
        self.tmpA /= Q
        self.A = decay * self.A + self.tmpA
        
        phi = phi0.copy()
        obj, dphi = self.objective(phi)

        for iter in range(self.maxiter):
            oldobj = obj            
            if self.trace:
                print '[%d] Objective: %g' % (iter, obj)

            for k in range(max(0, self.k0-self.backstep), self.maxarmijo):
                # determine gradient step
                alpha = self.beta**k * self.s
                phik = phi - alpha * dphi
                
                # project onto constraints
                phik /= vnorm(phik)

                # check Armijo condition
                objk, dphik = self.objective(phik)
            
                armijo = obj - objk >= self.sigma * np.sum(dphi * (phi - phik))

                if armijo:
                    if self.trace: print 'Armijo passed on step %d' % k
                    self.k0 = k
                    oldobj = obj
                    obj = objk
                    phi = phik.copy()
                    dphi = dphik.copy()
                    break

            if abs(obj/oldobj - 1) < self.otol:
                break

        if self.debug:
            angle = angles(phi0, phi)
            x = (obj, np.max(angle), np.min(angle), np.mean(angle), np.std(angle))
            debug('obj = %f, dphi max=%f, min=%f, mean=%f, std=%f degrees' % x)
        
        self.t += 1
        phi0[:] = phi
        
    def step(self, phi0, a, X):
        if self.rank == self.root:
            self.update(phi0, a, X)
        MPI.COMM_WORLD.Bcast(phi0, self.root)


