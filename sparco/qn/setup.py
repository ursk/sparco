from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os


ext_params = {
    'include_dirs': [numpy.get_include(), os.getenv('BLASPATH')],
    'library_dirs': [os.path.join(os.path.split(os.getenv('BLASPATH'))[0], 'lib')]
  }

ext_modules = [
    Extension('quasinewton', sources=['quasinewton.pyx'],
                  **ext_params),
        Extension("tokyo", sources=["tokyo.pyx"],
                  libraries=['cblas'],
                  **ext_params)
        ]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_modules)
    )
