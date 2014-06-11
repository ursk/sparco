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

There is a bug with Anaconda that it looks for mpi stuff in the folder /Anaconda1Anaonda2Anaconda3, the easiest way to solve this is with a simlink

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
- if this does not work, set cblas_include manually in the setup.py file



mpi4py
-----------
a working MPI installation is easiest to get with Anaconda python. Run 
conda install mpi4py
and it should install both mpi4py and the required openMPI backend.


running on an example data set
-----------
(this section is heavily work in progress)
- place data in filename0.h5 format in the ./data directory
- create sparco/out directory if it does not exist
- edit config.py and make sure all paths point to the right place ...
- data in h5 file is expected to be in the dataset /data and in time*channels format
- python run.py -m ecog -p test
- prefix (p) determins the model name, i.e. directory where output is saved
- model (m) determins which kind of model to run, i.e. gautam's lfp data or ecog data

example mpi call:
- /Users/urs/anaconda/bin/mpirun -np 3 /Users/urs/anaconda/bin/python run.py -m ecog -p test_mpi3
- 

Setting parameters
--------------
kwargs gets passed to datadb and contains parameters pertaining to data handling
- channels: subset of channels to use. does not have to be contigous, e.g. pick 'good' channels
- filenames: list of full paths to all the data files (if more than one, named sequentially, as in config.py)
- ...


p
