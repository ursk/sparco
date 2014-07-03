"""
Calls cython quasinewton algorithm with batches
"""
import numpy as np
import quasinewton as qn

class Objective(object):

    def __init__(self, phi, X, mask=None):
        self.phi = phi
        self.X = X
        self.mask = mask
        self.T = X.shape[-1]
        self.C, self.N, self.P = phi.shape
        self.alen = self.T+self.P-1
        self.xhat = np.empty((self.C, self.T))
        self.dx = np.empty_like(self.xhat)
        self.indx = 0

    def objective(self, x, df):
        """Return objective and modify derivative"""
        self.xhat.fill(0.)
        a = x.reshape(self.N, self.alen)        
        deriv = df.reshape((self.N, self.alen))
        deriv.fill(0.)        

        for b in range(self.P):
            self.xhat += np.dot(self.phi[:,:,b], a[:,b:b+self.T])

        # self.dx[:] = self.xhat - self.X[self.indx]
        self.dx[:] = self.xhat - self.X
        fx = 0.5 * (self.dx**2).sum()

        for b in range(self.P):
            deriv[:,b:b+self.T] += np.dot(self.phi[:,:,b].T, self.dx)

        if self.mask is not None:
            deriv *= self.mask

        return fx

def sparseqn_batch(phi, X, lam=1., maxit=25,
                   positive=False, Sin=None, debug=False,
                   delta=0.01, past=6, mask=None):
    """
    phi       - basis
    X         - array of batches
    maxit     - maximum quasi-newton steps
    positive  - positive only coefficients
    Sin       - warm start for coefficients
    debug     - print debug info
    delta     - sufficient decrease condition
    past
    mask      - an array of dims of coefficients used
                to mask derivative

    Sin is not modified
    """
    C, N, P = phi.shape
    npats = X.shape[0]
    T = X.shape[-1]
    alen = T + P - 1

    # instantiate objective class
    obj = Objective(phi, X, mask)

    # warm start values
    if Sin is not None:
        A = Sin.copy()
    else:
        # A = np.zeros((npats, N, alen))
        A = np.zeros((N, alen))

    lam = lam * np.ones(N*alen)

    # don't regularized coefficients that are masked out
    if mask is not None: lam *= mask.flatten()

    q = qn.owlbfgs(obj, N*alen, lam=lam,
                   debug=debug, maxit=maxit, delta=delta,
                   past=past, pos=positive)

    A = q.run(A.flatten()).reshape(N, alen)
    return A
