from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os


ext_params = {
    'include_dirs': [numpy.get_include(), os.getenv('BLASPATH')],
  }
libpath = os.path.join(os.path.split(os.getenv('BLASPATH'))[0], 'lib') 
if os.path.exists(libpath):
  ext_params['library_dirs'] = [libpath]

ext_modules = [
    Extension('quasinewton', sources=['quasinewton.pyx'],
                  **ext_params),
        Extension("tokyo", sources=["tokyo.pyx"],
                  # libraries=['lapack', 'f77blas', 'atlas', 'cblas'],
                  **ext_params)
        ]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
    )
