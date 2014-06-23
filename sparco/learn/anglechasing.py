from sparco.learn.learner import Learner

class RootSpikenet(Spikenet):

  def iteration(self):
    self.rootx = self.db.get_patches(self.bs)
    super(RootSpikenet, self).iteration()
    if self.write_interval and self.t % self.write_interval == 0:
      self.update_statistics()
      self.write()

  @time_track
  def infer_coefficients(self):
    super(RootSpikenet, self).infer_coefficients()

  @time_track
  def update_basis1(self):
    super(RootSpikenet, self).update_basis1()
    self.update_eta_and_phi()

  @time_track
  def update_basis2(self):
    super(RootSpikenet, self).update_basis1()
    self.rootdx = np.array([self.compute_dx(self.rootA[i]) for i in range(self.bs)])
    self.rootE = np.array([self.compute_E(self.rootdx[i]) for i in range(self.bs)])
    self.rootdphi = np.array([self.compute_dphi(rootdx[i])] for i in range(self.bs))
    self.update_eta_and_phi(dphi)

  def update_eta_and_phi(self, dphi):
    self.meandphi = np.mean(rootdphi, axis=0)
    self.meanE = np.mean(self.rootE)
    self.proposed_phi = self.compute_proposed_phi(self.phi, self.meandphi, self.eta)
    angle = self.compute_angle(new_phi)
    self.update_phi()
    self.update_eta()

class Learner(object):
  pass

class AngleChasing(Learner):
  """
  Try to keep basis phi (treated as single vector) changing
  a specified number of degrees on each step.
  """

  def update_phi(self):
    if self.phi_angle < self.phi_angle_threshold:
      do_center = self.centering_interval and self.t % self.centering_interval == 0
      self.phi = center(self.proposed_phi) if do_center else self.proposed_phi
      self.phi = smooth(self.phi) if smooth else self.phi
    else:
      print 'Update to phi too large. Rejecting.'

  def update_eta(self):
    if angle < self.target:
      self.eta *= self.angle_up_factor
    else:
      self.eta *= self.angle_down_factor

  # def __init__(self, obj, dims, eta=.0001, up=1.01, down=.99, target=5.,
  #        c=100, cmax=1, smooth=False, thresh=10):
  #   """
  #   obj      - returns E, dE/dphi
  #   dims       - C, N, P, T
  #   eta      - update rate
  #   up       - eta increase
  #   down       - eta decrease
  #   target     - target angle in degress of max angle change
  #   thresh     - discard spurious updates to basis
  #   c        - center basis functions every few steps
  #   cmax       - max shift in basis function on centering step
  #   """
  #   attributesFromDict(locals())
  #   self.C, self.N, self.P, self.T = dims
  #   if mpi.rank != mpi.root:
  #     return
  #
  #   self.debug = True
  #   self.t = 0
  #
  #   print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!this is learner with thresh", self.thresh

  # def update(self, phi, a, X):
  #   """
  #   Inputs
  #    phi   - current basis   (modified on return)
  #    a   - coefficients
  #    X   - data
  #   """
  #   oldphi = phi.copy()
  #   obj, dphi = self.obj(phi, a, X)
  #
  #   newphi = phi - self.eta * dphi
  #   newphi /= vnorm(newphi)
  #
  #   dot = np.sum(newphi[:]*oldphi[:])/ (np.linalg.norm(newphi) * np.linalg.norm(oldphi))
  #   angle = np.arccos(dot) * 180. / np.pi
  #   if np.isnan(angle): angle = 0.
  #   #angle = max(angles(phi, oldphi))
  #   if angle < self.thresh:
  #     phi[:] = newphi[:]
  #   else:
  #     print 'Update to phi too large. Rejecting.'
  #   if angle < self.target:
  #     self.eta *= self.up
  #   else:
  #     self.eta *= self.down
  #
  #   debug('[%d] eta = %g, angle = %g' % (self.t, self.eta, angle))
  #   self.t += 1


  # def step(self, phi, a, X):
  #   if mpi.rank == mpi.root:
  #     self.update(phi, a, X)
  #     if self.c is not None and self.t % self.c == 0:
  #       self.center(phi)
  #   mpi.bcast(phi, mpi.root)

  def update_rootdx(self):
    self.rootdx = 

  def update_xhat(self):
    self.xhat = reduce(np.add,
        (np.dot(self.phi[:,:,i], a[:,i:i+self.t]) for i in range(self.p)))

  def update_dx(self):
    self.dx =  self.x - self.xhat

  def update_E(self):
    self.E = 0.5 * np.linalg.norm(self.dx)**2

  def update_dphi(self):
    self.dphi = reduce(np.dstack,
        (np.dot(self.dx, a[i:i+self.t].T) for i in range(self.p)))

  def update_angle(self):
    num = np.sum(self.proposed_phi[:]*self.phi[:])
    denom = (np.linalg.norm(newphi) * np.linalg.norm(self.phi))
    angle = np.arccos(num/denom) * 180 / np.pi
    self.angle = 0 if np.isnan(angle) else angle

  def update_proposed_phi(self):
    proposed_phi = self.phi - self.eta * dphi
    self.proposed_phi = proposed_phi / sptools.vnorm(newphi)

  def update_phi(self):
    if self.angle < self.thresh:
      self.phi = self.proposed_phi
    else:
      print 'Update to phi too large. Rejecting.'

  def update_eta(self):
    if self.angle < self.target:
      self.eta *= self.up
    else:
      self.eta *= self.down

  def update_dx(self):
    self.dx = compute_dx(self.phi, self.a, self.x)

  def update_xhat(self):
    self.xhat = compute_xhat(self.phi, self.a)

  def update_E(self):
    self.E = compute_E(self.dx)

  def update_dphi(self):
    self.dphi = compute_dphi(self.dx, self.a)

  def update_proposed_phi(self):
    self.proposed_phi = compute_proposed_phi(self.dphi)

  def update_rootx(self):

  def update_eta(self):


# Non-OOP

  def compute_dx(x, a=None, phi=None, xhat=None):
    return x - (xhat if xhat else compute_xhat(phi, a))

# take matrices phi MxNxP and a Nx(P+T-1)
# convolve each vector along the P axis of phi with its correspondent along the (P+T-1) axis of a
# trim the partially overlapping convolution ends to yield an MxNxT matrix
# sum this along the N axis to yield MxT
  def compute_xhat(phi, a):
    p = phi.size[2]; t = a.size[1] - (p+1)
    return reduce(np.add, (np.dot(phi[:,:,i], a[:,i:i+t]) for i in range(p)))

  def compute_E(dx):
    return 0.5 * np.linalg.norm(dx)**2

  def compute_dphi(dx, a):
    t = dx.size[1]; p = a.size[2] - (t+1)
    return reduce(np.dstack, (np.dot(dx, a[i:i+t].T) for i in range(p)))

  def compute_angle(phi1, phi2):
    dot = np.sum(phi1*phi2) / (np.linalg.norm(newphi) * np.linalg.norm(self.phi))
    angle = np.arccos(dot) * 180 / np.pi
    return 0 if np.isnan(angle) else angle

  def compute_proposed_phi(phi, dphi, eta):
    newphi = phi - elf.eta * dphi
    return newphi / sptools.vnorm(newphi)

# Assorted Functions

  def center(arr, maxshift=None):
    """
    Shift each basis function to its center of mass by a maximum
    amount of cmax.

    Optionally, smooth basis functions.

    Center of mass is defined using sum of squares.
    """
    for n in range(self.N):
      s = np.sum(arr[:,n]**2, axis=0)
      total = np.sum(s)
      if total == 0.: continue
      m = int(np.round(np.sum(np.arange(self.P) * s)/total))
      shift = self.P/2 - m
      if maxshift:
        shift = np.sign(shift) * min(abs(shift), self.cmax)
      arr[:,n] = np.roll(arr[:,n], shift, axis=1)
      if shift > 0:
        arr[:,n,0:shift] = 0.
      elif shift < 0:
        arr[:,n,shift:] = 0.
      else:
        continue
      arr[:,n] /= np.linalg.norm(arr[:,n])
      print 'Shifting %d by %d' % (n, shift)

  def smooth()
    a = 1
    b = [0.25, .5, 0.25]
    for n in range(self.N):
      phi[:,n] = lfilter(b, a, phi[:,n], axis=1)

