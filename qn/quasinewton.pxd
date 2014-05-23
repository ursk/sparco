cimport numpy as np
cimport tokyo as blas

cdef extern from "string.h":
     void * memcpy(void *s1, void *s2, size_t n)
     void * memset(void *b, int c, size_t n)
     void bcopy(void *s1, void *s2, size_t n)
     double fmax(double x, double y)

cdef extern from "math.h":
     double fabs(double x)
     double sqrt(double x)

cdef inline double l1_norm(np.ndarray x):
    """Compute L1 norm of 1D vector"""
    return blas.dasum(x)

cdef inline unsigned long l0_norm(np.ndarray x):
    """Compute L0 norm of 1D vector"""
    cdef unsigned int i, l0 = 0
    for i in range(x.size):
        if x[i] != 0.: l0 += 1
    return l0

cdef inline void vncopy(np.ndarray x, np.ndarray y):
    """
    Set y = -x
    """
    blas.dvsetzero(y)
    blas.daxpy(-1., x, y)

cdef inline void vdiff(np.ndarray x, np.ndarray y, np.ndarray z):
    """
    z = x-y
    """
    blas.dcopy(x, z)
    blas.daxpy(-1., y, z)
    

cdef inline void l1_project(np.ndarray x, np.ndarray sign, np.ndarray lam):
    """
    Project x onto desired sign vector. 

    Modifies x
    """
    cdef unsigned int i
    cdef double *xp = <double*>x.data
    cdef double *signp = <double*>sign.data
    cdef double *lamp = <double*>lam.data        
    
    for i in range(x.size):
        if lamp[i] != 0. and xp[i] * signp[i] <= 0.:
            xp[i] = 0.

cdef inline void l1_nproject(np.ndarray x, np.ndarray sign, np.ndarray lam):
    """
    Project x onto desired negative of sign vector. (l1_project(x,-sign,lam))

    Modifies x
    """
    cdef unsigned int i
    cdef double *xp = <double*>x.data
    cdef double *signp = <double*>sign.data
    cdef double *lamp = <double*>lam.data        
    
    for i in range(x.size):
        if lamp[i] != 0. and xp[i] * signp[i] >= 0.:
            xp[i] = 0.

cdef inline void orthant(np.ndarray x, np.ndarray wp, np.ndarray pdf):
     """
     Set sign vector to choose orthant in negative projected gradient direction
     """
     cdef double *xp = <double*>x.data
     cdef double *wpp = <double*>wp.data
     cdef double *pdfp = <double*>pdf.data

     cdef unsigned int i
     for i in range(x.size):
          if xp[i] == 0.:
               wpp[i] = -pdfp[i]
          else:
               wpp[i] = xp[i]

cdef inline void positive_orthant(np.ndarray x, np.ndarray wp, np.ndarray pdf):
     """
     Set sign vector to choose orthant in negative projected gradient
     direction but not anywhere other than the positive orthant
     """
     cdef double *xp = <double*>x.data
     cdef double *wpp = <double*>wp.data
     cdef double *pdfp = <double*>pdf.data

     cdef unsigned int i
     for i in range(x.size):
          if xp[i] == 0. and pdfp[i] < 0.:
               wpp[i] = 1.
          else:
               wpp[i] = xp[i]

cdef inline void l1_gradient(np.ndarray x, np.ndarray g,
                             np.ndarray pg, np.ndarray lam):
    """
    Compute pseudo gradient given gradient for loss (excluding l1 term).

    Modifies pg.
    """
    cdef unsigned int i
    cdef double *xp = <double*>x.data
    cdef double *gp = <double*>g.data
    cdef double *pgp = <double*>pg.data
    cdef double *lamp = <double*>lam.data        

    for i in range(x.size):
        if lamp[i] == 0.:
            pgp[i] = gp[i]
        elif xp[i] < 0.:
            pgp[i] = gp[i] - lamp[i]
        elif xp[i] > 0.:
            pgp[i] = gp[i] + lamp[i]
        else:
            if gp[i] < -lamp[i]:
                pgp[i] = gp[i] + lamp[i]
            elif gp[i] > lamp[i]:
                pgp[i] = gp[i] - lamp[i]
            else:
                pgp[i] = 0.
    

cpdef inline double l1_penalty(np.ndarray x, np.ndarray lam):
    """
    Compute L1 penalty
    """
    cdef double *xp = <double*>x.data
    cdef double *lamp = <double*>lam.data
    cdef double l1 = 0.
    cdef unsigned int i
    
    for i in range(x.size):
        l1 += lamp[i] * fabs(xp[i])
    return l1
