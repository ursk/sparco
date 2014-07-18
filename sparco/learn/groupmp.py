class GroupMP(AngleChasing):

  def __init__(self, obj, dims, eta=.0001, up=1.01, down=.99, target=5.,
         c=100, cmax=1, extra=None):
    super(GroupMP, self).__init__(obj, dims, eta, up, down, target, c, cmax)
    self.extra = extra
    self.gsize = extra['gsize']
    self.orthonormal = extra['orthonormal']
    self.groups = self.N / self.gsize

  def update(self, phi, a, X):
    """
    Inputs
     phi   - current basis   (modified on return)
     a   - coefficients
     X   - data
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
