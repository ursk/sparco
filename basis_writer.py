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


class BasisWriter(object):

  def __init__(self, output_path=None, prefix=None, is_movie=False,
      create_plots=True):
    """
    prefix  - prefix to output files (and directories)
    output_path  - root path of output
    is_movie   - if input data is natural scenes, modify output
    create_plots   - whether or not to save plots
    """
    for k,v in locals().items():
      setattr(self, k, v)
    self.variance = None

    # create directories if necessary
    try:
      os.makedirs(os.path.join(self.output_path, self.prefix))
      if self.create_plots:
        for dir in ['unsorted', 'sorted', 'line', 'reconstruction', 'basis']:
          os.makedirs(os.path.join(self.output_path, self.prefix, dir))
    except OSError:
      pass

  def code(self, t):
    """ Generate string for output filenames.

    Args:
      t   - time step
    """
    code = ('%s-it=%06d,C=%d,N=%d,P=%d,T=%d,lam=%.2f,bs=%d,proc=%d') % (
        self.writer_settings['prefix'], t, self.C, self.N, self.P, self.T,
        self.inference_settings['lam'], self.bs, self.procs)
    return code
    def write(self, phi, A, iteration, code, error,
              cmap=plt.cm.jet, X=None, Xhat=None):
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
        # reorder indices as (basis number, batch * time)
        coeff = A.transpose(1,0,2).copy()
        coeff.shape = (coeff.shape[0], coeff.shape[1]*coeff.shape[2])
        
        # get norms and variance of coefficients
        l0norm = (coeff != 0.).sum(axis=1).astype('float64')
        l0norm /= np.prod(coeff.shape[1:])
        l0 = np.mean(l0norm)

        l1norm = np.abs(coeff).sum(axis=1)
        l1norm /= max(l1norm)

        l2norm = norm(phi)
        l2norm /= max(l2norm)

        # sort based on accumulated variance but display recent batch variance
        variance = np.var(coeff, axis=1)
        if self.variance is None:
            self.variance = variance
        else:
            self.variance += variance
        variance /= max(variance)
        order = np.argsort(self.variance)[::-1]

        # filenames for output
        unsorted = os.path.join(self.output_path, self.prefix, 'unsorted',
                                'basis-%s.png' % code)
        sorted = os.path.join(self.output_path, self.prefix, 'sorted',
                              'sorted-basis-%s.png' % code)
        lineplt = os.path.join(self.output_path, self.prefix, 'line',
                               'line-basis-%s.png' % code)
        reconplt = os.path.join(self.output_path, self.prefix, 'reconstruction',
                               'reconstruction-%s.png' % code)
        
        title='%05d l0:%0.3f e:%0.3f' % (iteration, l0, error)

        params = (l0norm, l1norm, variance, l2norm)
        sparams = (l0norm[order], l1norm[order], variance[order], l2norm[order])
        if self.plots:
            if self.is_movie:
                natmov_basis(phi, title, unsorted, params=params, figno=2)
                natmov_basis(phi[:,order], title, sorted, params=sparams, figno=3)
            else:
                display_basis(phi, title, unsorted, cmap=cmap, params=sparams, figno=2, display=True)
                display_basis(phi[:,order], title, sorted, cmap=cmap, params=sparams, figno=3, display=True)
                #line_plot(phi[:,order], lineplt, figno=4)
                #line_plot(phi, lineplt, figno=4)                                        

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


def natmov_reconstruction(X, Xhat, filename=None, display=False,
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


def plot_reconstruction(A, X, Xhat, filename=None, gain=.5, figno=32):
    """
    Plot original and recovered signal with coefficients as matrices
    """
    plt.figure(figno, figsize=(10,6))
    plt.clf()
    plt.ioff()
    imrange = gain * np.max(np.abs(X))

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


def natmov_basis(phi, title=None, filename=None, cmap=plt.cm.gray,
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


def sparse_movie(phi, X0, X, A0, A, name, movdir, fps=5, interval=1):
    """
    Make movie of sparsification

    Input:
     phi    - kernel (c, n, p)
     X0     - original signal
     X      - X0 + noise
     A0     - original causes
     A      - steps of causes
     name   - PNG output name
     movdir - where to dump PNGs
     interval - how many steps to skip
    """
    plt.figure(figsize=(8,8))
    plt.clf()
    plt.ioff()
    rows = 5
    cols = 2

    for i in range(0,len(A),interval):
        sp = 1        
        plt.clf()
        Xhat = np.array([correlate(phi[j], A[i], mode='valid').squeeze()
                         for j in range(len(phi))])
        xrng = np.max(np.abs(X))
        arng = np.max(np.abs(A0))

        plt.subplot(rows,cols,sp)
        plt.imshow(X0, vmin=-xrng, vmax=xrng, origin='upper')
        plt.title('Original')
        sp +=1
        plt.subplot(rows,cols,sp)
        plt.imshow(Xhat, vmin=-xrng, vmax=xrng, origin='upper')
        plt.title('Estimated')
        sp += 1
        plt.subplot(rows,cols,sp)
        plt.imshow(X, vmin=-xrng, vmax=xrng, origin='upper')
        plt.title('Noisy')
        sp += 1        
        plt.subplot(rows,cols,sp)
        plt.imshow(X0-Xhat, vmin=-xrng, vmax=xrng, origin='upper')
        plt.title('Reconstruction')
        sp += 1        
        plt.subplot(rows,cols,sp)
        plt.imshow(A0, vmin=-arng, vmax=arng, aspect=4, origin='upper')
        plt.title('True Coeff')
        sp += 1        
        plt.subplot(rows,cols,sp)
        plt.imshow(A[i], vmin=-arng, vmax=arng, aspect=4, origin='upper')
        plt.title('Estimated Coeff')

        sp += 1        
        plt.subplot(rows,cols,sp)
        plt.plot(A0.reshape(-1))
        plt.ylim([-arng, arng])
        plt.xlim([0, A0.size])
        plt.title('True Coeff')
        sp += 1        
        plt.subplot(rows,cols,sp)
        plt.plot(A[i].reshape(-1))
        plt.ylim([-arng, arng])
        plt.xlim([0, A[i].size])
        plt.title('Estimated Coeff')

        sp += 1        
        plt.subplot(rows,cols,sp)
        plt.scatter(A0.reshape(-1), A[i].reshape(-1))
        plt.title('True vs Estimated Coeff')
        
        #plt.subplots_adjust(hspace=0.60)
        plt.draw()        
        plt.savefig(os.path.join(movdir, '%04d_%s.png' % (i, name)))

    plt.ion()
    import subprocess
    command = ('mencoder',
               'mf://%s/*.png' % movdir,
               #'-vf', 'scale=800:-10',
               '-mf', 'fps=%d' % fps,
               '-ovc', 'lavc',
               '-lavcopts', 'vcodec=mpeg4',
               '-o', os.path.join(movdir, '%s.avi' % name))
    print 'Executing: %s ' % (' '.join(command))
    subprocess.check_call(command)
