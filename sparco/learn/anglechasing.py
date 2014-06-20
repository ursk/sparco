from sparco.learn.learner import Learner

class AngleChasing(Learner):
  """
  Jack's method, try to keep basis phi (treated as single vector) changing
  a specified number of degrees on each step.
  """

  def __init__(self, obj, dims, eta=.0001, up=1.01, down=.99, target=5.,
         c=100, cmax=1, smooth=False, thresh=10):
    """
    obj      - returns E, dE/dphi
    dims       - C, N, P, T
    eta      - update rate
    up       - eta increase
    down       - eta decrease
    target     - target angle in degress of max angle change
    thresh     - discard spurious updates to basis
    c        - center basis functions every few steps
    cmax       - max shift in basis function on centering step
    """
    attributesFromDict(locals())
    self.C, self.N, self.P, self.T = dims
    if mpi.rank != mpi.root:
      return

    self.debug = True
    self.t = 0

    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!this is learner with thresh", self.thresh

  def update(self, phi, a, X):
    """
    Inputs
     phi   - current basis   (modified on return)
     a   - coefficients
     X   - data
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
    if mpi.rank == mpi.root:
      self.update(phi, a, X)
      if self.c is not None and self.t % self.c == 0:
        self.center(phi)
    mpi.bcast(phi, mpi.root)

def obj(phi, a, x, T):
  dx = compute_dx(x, phi, a, T)
  E = compute_E(dx)
  dphi = compute_dphi(phi, a, dx, T)
  return E, dphi

def compute_dx(x, phi, a, T):
  xhat = np.zeros_like(x)
  for t in range(phi.shape[2]):
    xhat += np.dot(phi[:,:,t], a[:,t:t+T])
  return x - xhat

def compute_E(dx):
  return 0.5 * np.linalg.norm(dx)**2

def compute_dphi(phi, a, dx, T):
  dphi = np.zeros_like(phi)
  for t in range(phi.shape[2]):
    dphi[:,:,t] = np.dot(dx, a[t:t+T].T)
  return dphi

