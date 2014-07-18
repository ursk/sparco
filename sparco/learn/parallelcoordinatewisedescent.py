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
         t0=25, rho=15, memory=False, et=100, mpe=False):
    """
    obj      - returns E, dE/dphi (not used)
    dims       - C, N, P, T
    maxiter    - maximum coordinate sweeps
    tol      - tolerance in percentage change of objective
    epsilon    - initial diagonals of A
    t0       - annealing term
    rho      - annealing term, bigger more gradual start
    memory     - use epoch memory
    et       - epoch times
    """
    attributesFromDict(locals())
    self.C, self.N, self.P, self.T = dims
    if self.N % mpi.procs != 0:
      raise NotImplementedError('Basis functions not multiple of procs')
    if t0 < 2: raise ValueError('t0 < 2')
    self.Nl = self.N / mpi.procs

    # initialize A and B
    if mpi.rank == mpi.root:
      self.A = np.zeros((self.N, self.P, self.N, self.P))
      for i,j in np.ndindex((self.N, self.P)):
        self.A[i,j,i,j] = self.epsilon
      self.U = np.empty((self.N, self.P, self.P))              
      self.phi = np.empty((self.N, self.C, self.P))
      self.phiT = self.phi.transpose((1,0,2))  # a view of phi
      self.dphi = np.empty_like(self.phi)                  
      self.B = np.empty_like(self.phi)              
      self.Ac = np.empty((mpi.procs, self.Nl, self.P, self.Nl, self.P))
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
     a   - coefficients
     X   - data
    """
    with self.step_begin: pass

    with self.init:
      if mpi.rank == mpi.root:
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
      mpi.scatter(self.Ac, self.Al, mpi.root)
      mpi.scatter(self.dphi, self.dphil, mpi.root)

    # initialize local basis
    self.phil[:] = phi0[:,mpi.rank*self.Nl:(mpi.rank+1)*self.Nl].transpose((1,0,2))

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
        mpi.gather(self.phil, self.phi, mpi.root)

      with self.comp:
        if mpi.rank == mpi.root:
          obj = self.objective()
          self.stop[0] = int(abs(obj/oldobj - 1) < self.tol)
          oldobj = obj
      with self.broadcast:
        mpi.bcast(self.stop, mpi.root)
      if self.stop[0]: break

      # scatter derivatives to nodes
      with self.scatter:
        mpi.scatter(self.dphi, self.dphil, mpi.root)

    # broadcast all of phi to nodes before returning
    if mpi.rank == mpi.root:
      if self.debug:
        angle = self.angles(phi0, self.phiT)
        x = (self.t, obj, obj + self.norm, iter, decay, np.max(angle),
           np.min(angle), np.mean(angle), np.std(angle))
        debug(('[%d] obj-norm = %f, obj = %f, iter = %d, decay = %g, ' +
             'dphi max=%f, min=%f, mean=%f, std=%f degrees') % x)
      phi0[:] = self.phiT
    with self.broadcast:
      mpi.bcast(phi0, mpi.root)
    self.t += 1

    with self.step_end: pass
