from sparco.output.writer import Writer

class NaturalMovieWriter(Writer):

  def display_basis(phi, title=None, filename=None, cmap=plt.cm.gray,
                   params=None, figno=2, sb=(1,1,1), display=False):
      """
      Display natural movie basis
      """
      if params:
          l0norm, l1norm, variance, l2norm = params

      # reoder phi as (basis number, time, channel)
      phi = phi.transpose(1,2,0).copy()
      phi.shape = phi.shape[0:2]+2*(np.int(np.sqrt(phi.shape[2])),)
      n, t, d, d = phi.shape

      # create a matrix of basis vectors interleaved with black buffer lines
      buf = 3
      ncols = int(np.ceil(np.sqrt(n*t)/t))
      nrows = np.ceil(n/float(ncols))

      I = np.zeros((d*nrows + (nrows+1)*buf - 2, (t*d)*ncols + (ncols+1)*buf - 3))
      for i in range(n):
          patch = phi[i,::-1]
          sx = (d*t + buf) * (i % ncols) + buf - 1
          sy = (d + buf) * (i / ncols) + buf - 1
          prange = np.abs(patch).max()
          # rescale patch to [0, 1]
          if prange > .00001: patch = .5 + .5*patch/prange
          else: patch += .5
          for j in range(t):
              I[sy:sy+d, sx+d*j:sx+d*(j+1)] = patch[j]
          # add borders
          if params:
              I[sy-1, sx:sx + max(1,np.round(l1norm[i]*d*t))] = .6
              I[sy:sy + max(1,np.round(l0norm[i]*d)), sx-1] = .6
              I[sy+d, sx:sx + max(1,np.round(variance[i]*d*t))] = .6
              if l2norm is not None:
                  I[sy:sy+max(1,np.round(l2norm[i]*d)), sx+d*t] = 0.6           


      # plot the basis
      if display:
          fig = plt.figure(figno)
          plt.clf()
          ax = plt.subplot(*sb)
          plt.imshow(I, cmap=plt.cm.gray, interpolation='nearest', aspect='equal', origin='upper')
          plt.axis('off')
          if title: plt.title(title)
          plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.95,
                              wspace=0.02, hspace=0.02)
          plt.draw()
          
          # save figure    
          if filename:
              dpi = max(150, np.int(2*I.shape[0]/fig.get_figheight()))
              plt.savefig(filename, dpi=dpi)
      elif filename:
          plt.imsave(filename, I, cmap=plt.cm.gray)

  def plot_reconstruction(X, Xhat, filename=None, display=False,
                          figno=13, title='Batch reconstruction'):
    """
    Generate one plot with data and it's reconstruction
     X    - original
     Xhat - reconstruction
    """
    natmov_basis(X.transpose((1,0,2)), title=title,
                 figno=figno, sb=(2,1,1), display=display)
    natmov_basis(Xhat.transpose((1,0,2)), figno=figno, sb=(2,1,2),
                 filename=filename, display=display)
