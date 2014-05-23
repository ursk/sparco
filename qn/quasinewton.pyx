# Compiler directives
#cython: boundscheck=False
#cython: cdivision=False
#cython: infer_types=True
#cython: wraparound=False
#cython: nonecheck=False
#cython:profile
"""
This mostly a copy of liblbfgs by Naoaki Okazaki:

http://www.chokkan.org/software/liblbfgs/

liblbfgs uses SSE2 optimizations. These are replaced by BLAS.

[TODO] 1.3x slower than liblbfgs due to vector lambda
"""
import numpy as np
cimport numpy as np
import tokyo as blas
cimport tokyo as blas
from quasinewton cimport *

cdef class LineSearch(object):
    pass

cdef class BacktrackingLineSearch(LineSearch):
    """
    Backtracking line search, checks Armijo and strong Wolfe conditions.
    """
    cdef object func
    cdef unsigned int n
    cdef readonly np.ndarray xk, dfk
    cdef readonly double fk
    cdef public double eta
    cdef np.ndarray lam
    cdef bint pos

    cdef double armijo, wolfe, dec, inc
    cdef double min_step, max_step
    cdef unsigned int maxsteps
    cdef np.ndarray wp, dx

    def __cinit__(self, object func, unsigned int n, np.ndarray lam=None, bint pos=False):
        """
        func - function called as func.objective(x, df)        
        n    - dimensions
        lam  - L1 penalty
        """
        self.func = func
        self.n = n
        self.lam = lam
        
        self.xk = blas.dvnewempty(self.n)
        self.fk = 0.
        self.dfk = blas.dvnewempty(self.n)

        self.pos = pos

        if lam is not None:
            self.wp = blas.dvnewempty(self.n)
            self.dx = blas.dvnewempty(self.n)

        # linesearch params (p200, Nocedal and Wright, 1999)
        self.eta = 1.        
        self.armijo = 1e-4
        self.wolfe = 0.9
        self.dec = 0.5
        self.inc = 2.1
        self.min_step = 1e-20
        self.max_step = 1e20
        self.maxsteps = 20
        
    cdef bint wolfesearch(self, np.ndarray x, double f, np.ndarray df, np.ndarray p):
        """
        Line search with strong Wolfe conditions
        
        Inputs
         x     - start point        
         f, df - old function and derivative values
         p     - direction (-df modified by Hessian)

        Output
         Returns True if line search succeeded
         Internal variables are used from calling function

         For debugging only. Not used.
         
        [TODO] Haven't checked this code
        """
        cdef unsigned int i
        cdef double pkdfk
        cdef double pdf = blas.ddot(p, df)
        cdef double armijo = self.armijo * pdf
        cdef double wolfe = self.wolfe * pdf

        # check p is descent direction
        if pdf > 0: return False

        for i in range(self.maxsteps):
            blas.dcopy(x, self.xk)            
            blas.daxpy(self.eta, p, self.xk)
            self.fk = self.func.objective(self.xk, self.dfk)
            if self.fk - f <= self.armijo * self.eta:
                pkdfk = blas.ddot(p, self.dfk)
                if pkdfk > -wolfe:
                    self.eta *= self.dec
                elif pkdfk < wolfe:
                    self.eta *= self.inc
                else:
                    return True
            else:
                self.eta *= self.dec
            if self.eta < self.min_step or self.eta > self.max_step:
                return False
        return False

    cpdef bint search(self, np.ndarray x, double f, np.ndarray pdf, np.ndarray p):
        """
        Vanilla back-tracking line search suitable for L1 penalty
        
        Inputs
         x     - start point
         f, df - old function and derivative values
         pdf   - projected df
         p     - direction
        Output
         Returns True if line search succeeded
         Internal variables are used from calling function
        """
        cdef unsigned int i
        cdef double dxdf

        # positive only coefficients?
        if self.pos:
            positive_orthant(x, self.wp, pdf)
        else:
            orthant(x, self.wp, pdf)

        for i in range(self.maxsteps):
            # compute new step 
            blas.dcopy(x, self.xk)
            blas.daxpy(self.eta, p, self.xk)

            # project onto correct orthant            
            l1_project(self.xk, self.wp, self.lam)

            # evaluate new function and gradient values
            self.fk = self.func.objective(self.xk, self.dfk) + l1_penalty(self.xk, self.lam)

            # check sufficient decrease condition
            vdiff(self.xk, x, self.dx)
            dxdf = blas.ddot(self.dx, pdf)
            if self.fk - f <= self.armijo * dxdf:
                return True
            
            self.eta *= self.dec

        return False
    

cdef class QuasiNewton(object):
    pass
            
cdef class owlbfgs(QuasiNewton):

    cdef object func
    cdef LineSearch linesearch
    cdef unsigned int n, maxit, m
    cdef int start, end
    cdef double tol
    cdef public np.ndarray lam
    cdef unsigned int t
    cdef np.ndarray s, y, alpha, ss, sy, yy
    cdef bint debug
    cdef object log
    cdef bint pos
    cdef int past
    cdef np.ndarray pastf
    cdef double delta

    cdef np.ndarray x, df, p, pdf, ns, ny

    def __cinit__(self, object func, unsigned int n, 
                  double tol=0.00001, int past=0, double delta=0.01,
                  unsigned int maxit=100, np.ndarray lam=None, unsigned int m=10, 
                  bint pos=False, bint debug=False, object log=None):
        """
        Inputs
         func  - func.objective(x, df) returns value and sets df        
         n     - dimension of search vector
         tol   - tolerance of objective gradient
         maxit - maximum iterations
         debug - print debug output
         lam   - vector of sparsity parameters
         past  - (f[t-past]-f[t])/f[t] < delta
         delta 
         m     - size of ring buffer
         pos   - positive only coefficients
         log   - hdf5 file of steps (required debug == True)
        """
        self.func = func
        self.n = n
        
        self.tol = tol
        self.maxit = maxit
        if lam is None:
            self.lam = blas.dvnewzero(self.n)
        else:
            self.lam = lam
        self.pos = pos

        self.m = m
        self.debug = debug
        self.log = log

        self.past = past
        self.delta = delta
        self.pastf = blas.dvnewzero(past)

        self.linesearch = BacktrackingLineSearch(self.func, self.n, lam=self.lam, pos=self.pos)
        self.t = 0

        # allocate ring buffer
        self.s = blas.dmnewempty(self.m, self.n)
        self.y = blas.dmnewempty(self.m, self.n)
        self.alpha = blas.dvnewempty(self.m)
        self.start = 0
        self.end = -1
        
        # temporary dot products
        self.sy = blas.dvnewempty(self.m)
        self.yy = blas.dvnewempty(self.m)

        # optimization variables, allocate once
        self.x = blas.dvnewempty(self.n)
        self.df = blas.dvnewempty(self.n)
        self.p = blas.dvnewempty(self.n)
        self.pdf = blas.dvnewempty(self.n)                
        self.ns = blas.dvnewempty(self.n)
        self.ny = blas.dvnewempty(self.n)

        if log: raise NotImplementedError()

    def clear(self):
        """
        Reset optimization
        """
        self.t = 0
        self.start = 0
        self.end = -1

    cdef void update_direction(self):
        """
        Modify negative gradient direction using approximate inverse Hessian
        """
        cdef unsigned int stop, i
        cdef int j
        cdef double beta
        
        # request fast buffer access
        cdef np.ndarray[double, ndim=1] alpha = self.alpha, sy = self.sy, yy = self.yy
        cdef np.ndarray[double, ndim=2] s = self.s
        cdef np.ndarray[double, ndim=2] y = self.y

        # 2 step recursion for approximate inverse Hessian
        stop = min(self.t, self.m)
        j = self.end

        for i in range(stop):
            alpha[j] = blas.ddot(s[j], self.p) / sy[j]
            blas.daxpy(-alpha[j], y[j], self.p)
            j = (j + self.m - 1) % self.m

        if self.t > 0:
            blas.dscal(sy[self.end] / yy[self.end], self.p)

        j = self.start
        for i in range(stop):
            beta = blas.ddot(y[j], self.p) / sy[j]
            blas.daxpy(alpha[j] - beta, s[j], self.p)
            j = (j+1) % self.m
            
    cpdef np.ndarray unroll(self):
        """
        Recover the current inverse Hessian
        """
        cdef np.ndarray B, V
        cdef unsigned int stop, i, j
        cdef double rho
        
        if self.t == 0: raise ValueError()
        B = self.sy[self.end] / self.yy[self.end] * np.eye(self.n)
        stop = min(self.t, self.m)
        j = self.end
        for i in range(stop):
            rho = 1/self.sy[j]
            V = np.eye(self.n) - rho * np.outer(self.y[j], self.s[j])
            B = (np.dot(V.T, np.dot(B, V)) + rho * np.outer(self.s[j], self.s[j]))
            j = (j-1) % self.m
        return B

    cdef bint update_buffer(self):
        """
        Update ring buffer
        """
        cdef np.ndarray[double, ndim=1] sy = self.sy, yy = self.yy
        cdef double u, v

        u = blas.ddot(self.ns, self.ny)
        v = blas.ddot(self.ny, self.ny)
        
        if u == 0. or v == 0.:
            if self.debug: print 'Aborting update'
            return False
        self.end = (self.end + 1) % self.m
        if self.end == self.start and self.t > 0:
            self.start = (self.end + 1) % self.m
        blas.dcopy(self.ns, self.s[self.end])
        blas.dcopy(self.ny, self.y[self.end])
        self.sy[self.end] = u
        self.yy[self.end] = v
        return True

    cpdef run(self, np.ndarray x0):
        """
        Run optimization until maxit reached or gradient norm below tol
        """
        cdef double f
        cdef double rate
        blas.dcopy(x0, self.x)

        for self.t in range(self.maxit):
            # evaluate function plus penalty, get gradient of loss
            f = self.func.objective(self.x, self.df) + l1_penalty(self.x, self.lam)

            # compute pseudo-gradient 
            l1_gradient(self.x, self.df, self.pdf, self.lam)
            
            # quit if gradient norm sufficiently small
            if blas.dnrm2(self.pdf) / fmax(blas.dnrm2(self.x), 1.0) < self.tol: break

            # quit if sufficient decrease in f from past steps ago
            if self.past:
                if self.t > self.past:
                    rate = (self.pastf[self.t % self.past] - f) / f
                    if rate < self.delta: break
                self.pastf[self.t % self.past] = f

            # update gradient with inverse Hessian
            vncopy(self.pdf, self.p)
            if self.t == 0:
                self.linesearch.eta = 1./sqrt(blas.ddot(self.p, self.p))
            else:
                self.linesearch.eta = 1.
            self.update_direction()

            # constrain updated gradient to have same sign as -pseudo-gradient
            l1_nproject(self.p, self.pdf, self.lam)
            
            if self.linesearch.search(self.x, f, self.pdf, self.p):
                vdiff(self.linesearch.xk, self.x, self.ns)
                vdiff(self.linesearch.dfk, self.df, self.ny)
                blas.dcopy(self.linesearch.xk, self.x)                

                if not self.update_buffer(): break
                
                if self.debug:
                    l0norm = (self.x != 0.).sum().astype(np.float64)
                    print ' Iteration %d: fx = %f \t xnorm(2,1,0) = %f, %f, %d/%d \t gnorm = %f' % (self.t, f,
                                                              np.linalg.norm(self.x),
                                                              np.linalg.norm(self.x,1),
                                                              l0norm, self.x.size,
                                                              np.linalg.norm(self.linesearch.dfk)  )
            else:
                if self.debug: print 'Line search failed at step %d' % self.t
                break

        # return a copy as self.x may get used again
        return self.x.copy()
    
