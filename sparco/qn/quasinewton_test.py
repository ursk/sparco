import os
import numpy as np
import quasinewton
import quasinewton as qn
#import lbfgs
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
import h5py
from time import time as now
#import ipdb

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
    d.fill(0.)
    d += np.dot(A.T, (np.dot(A, x)-a))
    return fx

def f3(x, d, A, a):
    fx = 0.5 * ((np.dot(A, x) - a)**2).sum()
    d.fill(0.)
    d += np.dot(A.T, (np.dot(A, x)-a))
    return fx

def f2(x, A, a):
    fx = 0.5 * ((np.dot(A, x) - a)**2).sum()
    return fx

def df2(x, A, a):
    d = np.dot(A.T, (np.dot(A, x)-a))
    return d

def f4(x, A, a, lam):
    fx = 0.5 * ((np.dot(A, x) - a)**2).sum() + lam*np.sum(np.abs(x))
    d = np.dot(A.T, (np.dot(A, x)-a)) + lam*np.sign(x)
    return fx, d

def my_lasso():
    """ returns lasso objective and "gradient"
    """
    A, a = args
    fx = 0.5 * ((np.dot(A, x) - a)**2).sum()
    d.fill(0.)
    d += np.dot(A.T, (np.dot(A, x)-a))
    return fx
    


def power(n, p, diag=False):
    """
    Create ill-conditioned matrix with power-law eigenvalues
    """
    A = np.random.randn(n,n)
    if diag:
        Q = np.eye(n)        
    else:
        Q, R = np.linalg.qr(A)
    D = np.diag(np.sqrt(np.array([1./(i+1)**p for i in range(n)])))
    J = np.dot(Q, np.dot(D, Q.T))
    return J


def test(n):
    display = True if n == 2 else False
    a = np.random.randn(n)
    x = np.abs(np.random.randn(n)*3)
    A = np.random.randn(n,n)
    A = np.dot(A.T, A)
    A = np.eye(n) + A

    A = power(n, 2)
    #A = np.eye(n)
    
    print 'Eigenvalues: ', np.linalg.eigvals(A)

    args = (A, a)
    lam = .02
    m = 8
    maxit = 300
    obj = Obj(f, args)
#    q = qn.l1_penalty(obj, lam=lam * np.ones((n)))
    q = qn.owlbfgs(obj, n, lam=lam * np.ones((n)), debug=False, maxit=maxit, m=m, pos=False)
#   q = qn.owlbfgs(f, n, args=args, lam=lam * np.ones((n)), debug=True, maxit=maxit, m=m, stype='wolfe')
    x2 = x.copy()
    x3 = x.copy()
    tic = now()
    ret= q.run(x2)
    T1 = now() - tic

    #(ret, allvec) = q.run(x)
    
    b = np.dot(np.linalg.inv(A), a)
    print 'True min: ', f(b, np.zeros_like(x), (A, a))    
    print 'True argmin: ', b
    print 'Initial value: ', x
    print 'My solution: ', ret
    print 'Error: ', np.linalg.norm(ret-b)
    if False:
        print 'BFGS:'
        res = fmin_bfgs(f2, x, fprime=df2, args=args, full_output=1, retall=1)
        print res[0]
        print 'Error: ', np.linalg.norm(res[0]-b)    
        # print 'Hessian: ', np.linalg.inv(res[3])

    logfile = os.path.join(os.path.expanduser('~'), 'sn', 'py', 'spikes', 'qn', 'test.h5')

#     tic = now()
#     res2 = lbfgs.bfgsl1(f3, x3, lam = lam, args=args, debug=False, log=None)
#     t2 = now() - tic
#     print 'bfgsl1: ', t1, f4(res2, A, a, lam)[0], res2
#     print 'qn: ', t2, f4(ret, A, a, lam)[0], ret

#     bounds = ((0, None),) * n
#     tic = now()
#     x, _, _ = fmin_l_bfgs_b(f4, x3,args=(A,a,lam), bounds=bounds, maxfun=maxit)
#     t3 = now() - tic
#     print 'lbfgsb: ', t3, f4(x, A, a, lam)[0], x

#     if n < 10:
#         print 'A.T A: ', np.dot(A.T, A)
#         B = q.unroll()
#         print 'B^-1: ', np.linalg.inv(B)

    #h5 = h5py.File(logfile, 'r')
    #allvec5 = h5['x'][:]
    #h5.close()

    if False and display:
        plot_path(allvec, args, 1, f)
        #plot_path(allvec2, args, 2, f)

    input = raw_input()        
        

def plot_path(allvec, args, figure, f):
    plt.figure(figure)
    plt.clf()
    delta = 0.025
    xm = np.arange(-2, 2, 0.025)
    X, Y = np.meshgrid(xm, xm)
    df = np.zeros_like(allvec[0])
    Z = np.array([[f(np.array([i, j]), df, args=args) for i in xm] for j in xm])
    plt.contour(X, Y, Z)
    ax = plt.gca()
    ax.set_aspect('equal')

    for i in range(len(allvec)-1):
        px = [allvec[i,0], allvec[i+1, 0]]
        py = [allvec[i,1], allvec[i+1, 1]]
        plt.plot(px, py)

def generate_test_cases(cases=5):
    logfile = os.path.join(os.path.expanduser('~'), 'sn', 'py', 'spikes', 'qn', 'bigcases.h5')
    h5 = h5py.File(logfile, 'w')
    t = 0.
    
    for i in range(cases):
        n = 2000 + 0*np.random.randint(10, 1000)
        a = np.random.randn(n)
        x = np.random.randn(n)*3
        A = power(n, 2)
    
        args = (A, a)
        lam = np.random.rand()*.01
        m = 8
        maxit = 100

        tic = now()
        res = lbfgs.bfgsl1(f, x, lam = lam, args=args, debug=False, log=None)
        t += (now() - tic)
        
        subgroup = h5.create_group('case-%03d' % i)
        subgroup.create_dataset('A', data=A)
        subgroup.create_dataset('a', data=a)        
        subgroup.create_dataset('x', data=x)
        subgroup.create_dataset('r', data=res)
        subgroup.create_dataset('lam', data=np.array([lam]))

    print 'Computation time: ', t
    h5.close()
    
if __name__ == "__main__":
    #generate_test_cases(1)
    test(200)
