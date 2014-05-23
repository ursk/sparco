import os
import numpy as np
import quasinewton
import quasinewton as qn

try:
    import ipdb #works on mothership, not cerberus
except:
    pass

class Obj(object):

    def __init__(self, f, args):
        self.f = f
        self.args = args

    def objective(self, x, df):
        f = self.f(x, df, self.args)
        return f

def f(x, d, args):
    A, a = args
    fx = 0.5 * ((np.dot(A, x) - a)**2).sum()
    #d.fill(0.) # derivative vecrtors get updated,
    #d += np.dot(A.T, (np.dot(A, x)-a)) # not returned to avoid reallocating the vector. 
    d = np.dot(A.T, (np.dot(A, x)-a)) # breaks the update, need to fill and +=
    return fx


def test(n=10):
    """ 
    the model to be optimized here is a quadratic form
    f(x) = 1/2 (Ax-a)^2, the gradient is A'(Ax-a). Find parameters x given data A, a
    """
    a = np.random.randn(n)
    x = np.abs(np.random.randn(n)*3)
    A = np.random.randn(n,n)
    A = np.dot(A.T, A)
    A = np.eye(n) + A
    
    print 'Eigenvalues: ', np.linalg.eigvals(A)

    args  = (A, a) # parameters of the model to optimize: plug in PCA data here.
    lam   = .1 # this is a sparseness parameter?!
    m     = 8 #would increase this to the dimension of problem to 
    maxit = 300
    obj   = Obj(f, args) # creating an instantiation of 
    # to call qn.owLBFGS need to package the function to optimize in an object.
    q = qn.owlbfgs(obj, n, lam=lam * np.ones((n)),
                   debug=True, maxit=maxit, m=m, pos=True) # orthant wise LBFGS, dimensions, lambdas, iterations...
    ret = q.run(x) # q is a c function, no idea what this does

    b = np.dot(np.linalg.inv(A), a) # true minimumu
    print 'True min: ', f(b, np.zeros_like(x), (A, a))    
    print 'True argmin: ', b
    print 'Initial value: ', x
    print 'My solution: ', ret
    print 'Error: ', np.linalg.norm(ret-b)

if __name__ == '__main__':
    test()
