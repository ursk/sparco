from __future__ import print_function

from distutils.extension import Extension
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from Cython.Build import cythonize

import io
import os
import sys

import numpy
import os

import sparco

if os.path.getenv('BLASPATH') is None:
  raise Exception("You must set BLASPATH to the directory containing cblas.h")

here = os.path.abspath(os.path.dirname(__file__))

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.md', 'CHANGES.md')

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
		name='sparco',
		version='0.0.1',
		url='http://github.com/ursk/sparco/',
    license='MIT',
		author='Amir Khosrowshahi, Urs Koster, Sean Mackesey',
    author_email='s.mackesey@gmail.com',
    description='convolutional sparse coding implemented with openMPI',
    long_description=long_description,
    tests_require=['pytest'],
    install_requires=[
      'traceutil',
      'pfacets'
      ],
    ext_modules = cythonize(ext_modules),
		packages=[''],
    include_package_data=True,
    platforms='any',
)
