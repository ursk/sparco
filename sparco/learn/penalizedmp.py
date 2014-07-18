class PenalizedMP(AngleChasing):

  def __init__(self, obj, dims, eta=.0001, up=1.01, down=.99, target=5.,
         c=100, cmax=1, extra=None):
    super(PenalizedMP, self).__init__(obj, dims, eta, up, down, target, c, cmax)
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
     a   - coefficients
     X   - data
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
