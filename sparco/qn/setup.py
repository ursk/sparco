from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os


cblas_include = os.getenv('BLASPATH')
print cblas_include
include_dirs = [numpy.get_include(), cblas_include]
lib = os.path.join(os.path.split(os.getenv('BLASPATH')[0], 'lib'))
library_dirs = [lib]

ext_modules = [
        Extension('quasinewton',
                  sources=['quasinewton.pyx'],
                  depends=['quasinewton.pxd'],
                  include_dirs=[numpy.get_include(), cblas_include],
                  library_dirs=library_dirs,
                  libraries=[]),
        Extension("tokyo",
                  sources=["tokyo.pyx"],
                  depends=['tokyo.pxd'],
                  libraries=['cblas'], 
                  library_dirs=library_dirs,
                  include_dirs=include_dirs),   
        ]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
    )

