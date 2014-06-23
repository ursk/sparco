import os

# TODO necessary to use AGG?
import matplotlib
matplotlib.use('AGG', warn=False)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
import numpy as np
import h5py
from scipy.misc.pilutil import imsave
from scipy.signal import correlate


class Writer(object):

  def __init__(self, spikenet, output_path=None, is_movie=False,
      create_plots=True):
    """
    output_path  - root path of output
    is_movie   - if input data is natural scenes, modify output
    create_plots   - whether or not to save plots
    """
    for k,v in locals().items():
      setattr(self, k, v)
    self.variance = None

    # create directories if necessary
    try:
      os.makedirs(self.output_path)
      # if self.create_plots:
      #   for dir in ['unsorted', 'sorted', 'line', 'reconstruction', 'basis']:
      #     os.makedirs(os.path.join(self.output_path, self.prefix, dir))
    except OSError:
      pass

  # def code(self):
  #   """ Generate string for output filenames.
  #
  #   Args:
  #     spikenet   - spikenet
  #   """
  #   sn = self.spikenet
  #   code = 'it%06d_%dx%dx%dx%d_lam%.2f_bs%d_proc%d' % (
  #       sn.t, sn.C, sn.N, sn.P, sn.T,
  #       sn.inference_settings['lam'], sn.bs, sn.procs)
  #   return code

  def write_configuration(self, config):
    with f as open(os.path.join(self.output_path, 'config.txt')):
      pp = pprint.PrettyPrinter(indent=2, stream=f)
      pp.pprint(config)


  # TODO revise the arglist here
  # def write(self, phi, A, iteration, code, error,
  #           cmap=plt.cm.jet, X=None, Xhat=None, spikenet=None, t=None):
  def write_snapshot(self):
      """
      Output basis and plots to file
       phi       - basis
       A         - coefficients (batch, basis, time)
       iteration - integer
       code      - prefix for output
       error     - energy / energy with zero A
       X         - data
       Xhat      - reconstructed data
       l2norm    - l2norm of basis
       
      Edge bars correspond to:
             top: l1 norm
        left: l0 norm       right: basis norm
             bottom: variance
      """
      # TODO temp hack; make everything use the passed in spikenet
      # code = self.code(spikenet)

      # filenames for output
      # unsorted = os.path.join(self.output_path, self.prefix, 'unsorted',
      #                         'basis-%s.png' % code)
      # sorted = os.path.join(self.output_path, self.prefix, 'sorted',
      #                       'sorted-basis-%s.png' % code)
      # lineplt = os.path.join(self.output_path, self.prefix, 'line',
      #                        'line-basis-%s.png' % code)
      # reconplt = os.path.join(self.output_path, self.prefix, 'reconstruction',
      #                        'reconstruction-%s.png' % code)
      
      title='%05d l0:%0.3f e:%0.3f' % (iteration, l0, error)
      display_basi

      coefficient_statistics = self.compute_coefficient_statistics(self.rootA)
      sort_order = np.argsort(self.accumulated_basis_variance)[::-1]
      sorted_params = [ x[sort_order] for x in coefficient_statistics ]

      params = (l0norm, l1norm, variance, l2norm)
      sparams = (l0norm[order], l1norm[order], variance[order], l2norm[order])
      if self.plots:
        display_basis(phi, title, unsorted, cmap=cmap, params=sparams, figno=2, display=True)
        display_basis(phi[:,order], title, sorted, cmap=cmap, params=sparams, figno=3, display=True)

      # save basis to file (overwrite)
      basisf = os.path.join(self.output_path, self.prefix, 'basis',
                            'basis-%s.h5' % code)
      print "DEBUG: Trying to open file", basisf
      h5 = h5py.File(basisf, 'w')
      h5.create_dataset('phi', data=phi)
      h5.create_dataset('order', data=order)
      h5.create_dataset('variance', data=variance)
      h5.create_dataset('l0', data=l0norm)        
      h5.create_dataset('l1', data=l1norm)
      h5.create_dataset('l2', data=l2norm)
      h5.close()

      # create soft-link to basis file
      linkf = os.path.join(self.output_path, self.prefix, 'basis.h5')
      try:
          os.remove(linkf)
      except:
          pass
      os.symlink(basisf, linkf)

      # display reconstruction
      if self.plots and X is not None and Xhat is not None and A is not None:
          if self.is_movie:
              natmov_reconstruction(X, Xhat, reconplt, display=True)
          else:
              mx = 5
              plot_reconstruction(A[:mx], X[:mx], Xhat[:mx], reconplt)


def plot_reconstruction(

    npat = len(A)
    for i in range(npat):
        ax = plt.subplot(npat,3,i*3+1)
        plt.imshow(X[i], vmin=-imrange, vmax=imrange, origin='lower')
        #ax.set_xticks([])
        #ax.set_yticks([])
        ax.set_aspect('auto')
    
        ax = plt.subplot(npat,3,i*3+2)
        plt.imshow(Xhat[i], vmin=-imrange, vmax=imrange, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')

        ax = plt.subplot(npat,3,i*3+3)
        plt.imshow(A[i], origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')

    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98,
                        wspace=0.03, hspace=0.11)    
    plt.ion()
    plt.draw()    
    
    if filename is not None:
        plt.savefig(filename)

def plot_reconstruction_line(A, X, Xhat, filename=None):
    """
    Plot original and recovered signal, with coefficients as line plot
    """
    # plot original and recovered signal
    plt.figure(2, figsize=(10,6))
    plt.clf()
    plt.ioff()
    std = np.std(X)
    imrange = np.max(np.abs(X))
    arange = np.max(np.abs(A))
    scale = 10
    std = scale * np.std(A)

    colors = [colorConverter.to_rgba(i) for i in ('b','g','r','c','m','y','k')]
    linewidths = [.3]
    segs = np.empty((A.shape[1],A.shape[2],2))
    segs[:,:,0] = np.arange(A.shape[2])

    npat = len(A)
    for i in range(npat):
        ax = plt.subplot(npat,5,i*5+1)
        plt.imshow(X[i], vmin=-imrange, vmax=imrange, origin='lower')
        if i == 0: plt.title('Original signal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    
        ax = plt.subplot(npat,5,i*5+2)
        plt.imshow(Xhat[i], vmin=-imrange, vmax=imrange, origin='lower')
        if i == 0: plt.title('Reconstruction')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')

        ax = plt.subplot(npat,5,i*5+3)
        plt.imshow(X[i] - Xhat[i], vmin=-imrange, vmax=imrange, origin='lower')
        if i == 0: plt.title('Error')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')

        ax = plt.subplot(npat,5,i*5+4)
        plt.imshow(A[i], vmin=-arange, vmax=arange, origin='lower')
        if i == 0: plt.title('Coefficients')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    
        ax = plt.subplot(npat,5,i*5+5)
        if i == 0: plt.title('Coefficients')
        segs[:,:,1] = A[i]/std
        lines = LineCollection(segs, offsets=(0,1), colors=colors, linewidths=linewidths)
        ax.add_collection(lines)
        ax.set_xlim(0, A.shape[2]-1)
        ax.set_ylim(-.5, A.shape[1]+.5)    
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    
    plt.draw()
    plt.ion()
    if filename is not None:
        plt.savefig(filename)



def display_basis(phi, title=None, filename=None, cmap=plt.cm.jet, params=None,
                  ncols=None, figno=1, display=False, dims=None):
    """
    Display basis in channel, time format

    Plots are correlation kernels, not convolution kernels.
    """
    if params:
        l0norm, l1norm, variance, l2norm = params
                  
    # reoder phi as (basis number, channel, time)
    phi = phi.transpose(1,0,2).copy()
    phi = phi[:,:,::-1] # correlation kernel plot
    m, n, o = phi.shape

    # create a matrix of basis vectors interleaved with black buffer lines
    buf = 3
    if dims is None:
        if ncols == None:
            ncols = int(np.ceil(np.sqrt(m)))
            a = int(np.sqrt(m))
            if m > a*(a+1): nrows = a + 1
            else: nrows = a
        else:
            nrows = int(np.ceil(np.double(m) / ncols))
    else:
        nrows, ncols = dims

    I = np.zeros((n*nrows + (nrows+1)*buf - 2,
                  o*ncols + (ncols+1)*buf - 3))
    for i in range(m):
        # import ipdb
        # flip along channel axis (deep sites at bottom)
        patch = phi[i,::-1]
        sx = (o + buf) * (i % ncols) + buf - 1
        sy = (n + buf) * (i / ncols) + buf - 1
        prange = np.abs(patch).max()
        # rescale patch to [0, 1]
        if prange > .00001: patch = .5 + .5*patch/prange
        else: patch += .5
        I[sy:sy+n, sx:sx+o] = patch
        # add borders
        if params:
            I[sy-1, sx:sx + max(1,np.round(l1norm[i]*o))] = .6     # top
            I[sy:sy + max(1,np.round(l0norm[i]*n)), sx-1] = .6     # left
            I[sy+n, sx:sx + max(1,np.round(variance[i]*o))] = .6   # bottom
            if l2norm is not None:
                I[sy:sy+max(1,np.round(l2norm[i]*n)), sx+o] = .6            

    # plot the basis
    if display:
        fig = plt.figure(figno)
        plt.clf()
        plt.imshow(I, cmap=cmap, interpolation='nearest', aspect='equal',
                   origin='upper',vmax=1, vmin=0)
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
        imsave(filename, I)
    

def line_plot(phi, filename=None, figno=4, gain=1., display=False,
              linewidth=0.2, colors=True, dpi=300):
    """
    Display basis as line plots

    Correlation plots (not convolution).
    May have screwed up having channel 1 at bottom
    """
    # reoder phi as (basis number, channel, time)
    phi = phi.transpose(1,0,2).copy()
    phi = phi[:,:,::-1]
    m, n, o = phi.shape

    # create a matrix of basis vectors interleaved with black buffer lines
    buf = 3
    ncols = int(np.ceil(np.sqrt(m)))
    a = int(np.sqrt(m))
    if m > a*(a+1): nrows = a + 1
    else: nrows = a

    fig = plt.figure(figno)
    plt.clf()
    plt.ioff()
    
    segs = np.empty(phi.shape[1:] + (2,))
    offset = np.arange(phi.shape[1])[:, np.newaxis]    
    segs[:,:,0] = np.arange(phi.shape[2])

    std = 0.5/np.std(phi) * gain
    colors = [colorConverter.to_rgba(i) for i in ('b','g','r','c','m','y','k')]
    #colors = [colorConverter.to_rgba(i) for i in ('k')]    
    linewidths = [linewidth]
    for i in range(m):
        ax = plt.subplot(nrows, ncols, i+1)
        segs[:,:,1] = std* phi[i]
        col = colors[i % len(colors)]
        lines = LineCollection(segs, offsets=(0,1), colors=col, linewidths=linewidths)
        ax.add_collection(lines)
        ax.set_xlim(0, phi.shape[2]-1)
        ax.set_ylim(-2, phi.shape[1]+ 2)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.02, hspace=0.02)
        
    # plot the basis
    plt.ion()
    plt.draw()

    # save figure    
    if filename:
        plt.savefig(filename, dpi=dpi)
