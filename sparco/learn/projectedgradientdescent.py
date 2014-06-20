import numpy as np

import mpi

class ProjectedGradientDescent(Learner):
  """
  [TODO] Make code more efficient
  """
  
  def __init__(self, obj, dims, maxiter=20, tol=1e-8,
         epsilon=0.001, t0=2, rho=15,
         s=1, beta=.5, sigma=1e-4):
    """
    obj      - returns E, dE/dphi
    dims       - C, N, P, T
    s, beta, sigma - Armijo params (see Bertsekas)
    maxiter    - maximum number of steps in line search
    """
    attributesFromDict(locals())
    self.C, self.N, self.P, self.T = dims

    if mpi.rank != mpi.root:
      return
    
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
     a    - coefficients
     X    - data

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
    if mpi.rank == mpi.root:
      self.update(phi0, a, X)
    mpi.bcast(phi0, mpi.root)


