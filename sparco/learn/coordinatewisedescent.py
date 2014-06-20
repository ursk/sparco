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
         t0=10, rho=15):
    """
    obj      - returns E, dE/dphi (not used)
    dims       - C, N, P, T
    maxiter    - maximum coordinate sweeps
    tol      - tolerance in percentage change of objective
    epsilon    - initial diagonals of A
    t0       - annealing term
    rho      - annealing term, bigger more gradual start
    memory     - use epoch memory
    """
    attributesFromDict(locals())
    self.C, self.N, self.P, self.T = dims
    if t0 < 2: raise ValueError('t0 < 2')
    if mpi.rank != mpi.root:
      return

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
     a   - coefficients
     X   - data
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
    if mpi.rank == mpi.root:
      self.update(phi, a, X)
    mpi.bcast(phi, mpi.root)
