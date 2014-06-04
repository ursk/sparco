sparco - Convolutional Sparse Coding
======







Installation
------------
1. uses a custom quasinewton package that depends on "tokyo" for BLAS bindings.
2. uses mpi4py which requires openMPI to be installed
   (on my machine, need /opt/local/bin/ipython-2.6 )


Package structure:
------------------

run.py 
has a function def for each data set (climate, LFP...) where the inference and learning
method is specified, all the parameters like file names, dimensionalities etc. It then creates the dataset with a call to imageDB.ClimateDB and runs the algorithm with a call to sp.spikenet

datadb.py
loads 'data', 'mean', and 'var' from a single or a set of hdf5 files. 
creates self.dset


learner.py
leraning the basis function using 

sp.py 
sparse coding objective function and wrapper for learner

sptools.py
saves learning progress. Images of basis functions and sparsified data. 


Notes for usage:

- if mean and var are supplied, mean^2 has to be smaller than var
- if mean and var are not supplied, the data is assumed to be normalized





Running jobs on hopper:
-----------------------

module list
module swap PrgEnv-pgi PrgEnv-gnu
module load python acml cython mpi4py 
module show acml
ls /opt/acml/5.3.1/pgi64/include/acml.h 
ls /opt/acml/5.3.1/open64_64/includes

you can place data in:
/project/projectdirs/vacet/mantissa/urs
cd $SCRATCH




Running MPI jobs:
-----------------------
The easiest way to get this to work is with Anaconda python. 
Conda install mpi4py installs not only the python package but also the mpi backend. 
Run jobs like
/Users/urs/anaconda/bin/mpirun -np 4 /Users/urs/anaconda/bin/python parallel_sparsify.py 


Notes on qn
-----------
quasinewton:
Amir's implementation of orthant-wise lbfgs with L1 constraint

tokyo: 
cython blas wrapper that's used by quasinewton, see https://github.com/tokyo/tokyo

to build the modules: 

- find cblas.h on your system (e.g. using the locate function)
- set $BLASPATH to point to the directory cblas.h is in, e.g.
export BLASPATH="/System/Library/Frameworks/vecLib.framework/Versions/A/Headers/"
- make all