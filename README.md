# sparco

## Installation

This code depends on `numpy`, `h5py`, and `mpi4py`. The command-line steps below do not cover installation of these packages. One way to gain access to these packages is to install [the Anaconda distribution](http://continuum.io/downloads) of Python, which includes all of the above as well as many other packages useful in scientific computing.

One of the libraries used by this code must be linked against BLAS. If your `cblas.h` file is located in a nonstandard location, then you must set the environment variable BLASPATH to the directory containing `cblas.h` before installation.

    git clone --recursive https://github.com/ursk/sparco
    export BLASPATH=/path/to/my/dir/containing/cblas.h  # optional
    cd sparco/lib/quasinewton
    setup.py build_ext --in-place

The `--recursive` option is used because dependency packages `pfacets`, `traceutil`, and `qn` have been included as git submodules. This is because this code may be run on systems where the user is not able to install packages.

## Usage

The codebase has been designed to support the calling of CSC in multiple contexts, allowing the ability to include CSC as part of a larger data-processing pipeline. However, a default entry point for the code is provided by the `csc` script. This script allows convolutional sparse coding to be carried out (with flexible configuration) on a single dataset.

The `csc` script builds a configuration dictionary. This dictionary is built from two sources:

- command line options
- a local configuration file, which must either reside in the user's home directory or have its location passed as a command line option

Given the presence of this script on the path, a call to `csc` might look like:

    ❯ csc -i /path/to/dir/with/my/h5/files -o /path/to/output/directory \
    -C /path/to/config/file

The input path (`-i`), is expected to be a directory containing one or more `.h5` files at the top level. The output path (`-o`) can be an arbitrary directory. All directories in the output path that do not already exist will be arbitrarily created. Details on the format of the config file pointed to by `-C` can be found in [](#configuration-file-structure). `csc` will run until it has hit the configured number of iterations. As of yet, there is no way to quit cleanly-- a kill/interrupt signal must be used (Ctrl-C on Unix).

Assuming `mpirun` is in `$PATH`, to run the code over mpi with, say, 4 processes:

    ❯ mpirun -n4 csc command-line-options...

## Configuration

@TD are defaults for ALL params rly specified in the docstrings

While parameters are specified differently depending on whether the command-line or a configuration file is used, ultimately they are mapped to keyword arguments of the `__init__` method of a class. Thus the default values and documentation for all parameters are specified in the source code in the `__init__` method and corresponding docstring for each class. All user-specified configuration overrides defaults, with command-line options overriding the configuration file.

A thorough understanding of how to configure the code is best gained by reading two sources:

- the output of `csc --help`, which provides a description of all available command line options. (The same information is available in the `ArgParse` specification of the `csc` source)
- the sample configuration file `sample_config.py`. For convenience, this file contains specifications for all possible parameters as well as their corresponding documentation. In practice, it is not necessary to specify so many parameters for most use cases, since the defaults are sufficient.

### Required Parameters

There are some parameters for which no reasonable defaults can be defined. Thus, the following parameters *must* be provided to `csc` through either command line options or the configuration file:

@TD required config params

### Configuration File Structure

A configuration file is just a python module that defines a dictionary `config`. This dictionary should hold other dictionaries which contain keyword arguments for the `__init__` methods of the various classes described above. The structure of `config` is best understood by looking at the sample configuration file provided as sample_config.py, but a brief description is provided here. `config` should contain the keys:

@TD review to make sure these are the only keys
- `'sampler'`: dict of keyword arguments for `sparco.sampler.Sampler#__init__`
- `'nets'`: an array of dicts, each having keyword arguments for `sparco.sp.Spikenet#__init__`
- `'trace'`: a dict that may contain keys:
    - `'RootSpikenet'`: dict of keyword arguments for `sparco.trace.sp.RootSpikenet#__init__`
    - `SparseCoder`: dict of keyword arguments for `sparco.trace.sparse_coder.SparseCoder#__init__`

## Architecture

This codebase is modular. The top-level division of code is between *core*, *data-loading*, and *output* classes.

### Core

Core code implements convolutional sparse coding within RAM, without reference to the source or meaning of the data used as input. Data is obtained for each algorithm iteration through a single call to an data provider object (see [](#data-loading)).

The major steps in the CSC algorithm are "learning" (given a dataset, generation of a basis) and "inference" (given a dataset and a basis, generation of coefficients). They are implemented in separate modules that plug in to a central class `Spikenet`. `Spikenet` manages the algorithm's central loop. Over the course of this loop, algorithm parameters are held constant. It is sometimes desirable to run the algorithm several times with different configurations arranged in a serial pipeline; the output of one segment is used as the input to the next. A class `SparseCoder` implements this functionality by managing a metaloop that spins up a sequence of `Spikenet` instances.

### Data-Loading

The core accesses data by calling the `get_patches` method of a "sampler" object. In principle, any object that responds to `get_patches` with a 3d array can be used in place of an instance of the included `Sampler` class.

`Sampler` wraps a single HDF5 file or directory containing HDF5 files. It caches a random subset of the wrapped (2d) data in memory; `get_patches` returns a set of random fragments of this cache in the form of a 3d array. The cached subset is refreshed once a configured number of patches have been drawn from it.

### Output

In the interest of clean source code and composability of core classes, all logging, visualization, and writing of intermediate and final results to disk have been implemented in a separate software layer located in `sparco.trace`. This layer may be configured (or disabled) independently of core or data-loading components. The module structure of `sparco.trace` parallels that of `sparco`-- for each class in `sparco` for which output is desired, a corresponding `Tracer` class is implemented in `sparco.trace`. These `Tracer` classes inherit from the `Tracer` class defined in the external `traceutil` package.

`traceutil` provides a framework for applying nested decorators, logging, and capturing profiling data and state snapshots of an evolving object. This provides for complex configuration of output and periodic capture of algorithm state without cluttering the core source with output-specific meta-parameters and conditional statements. Individual `traceutil.Tracer` subclasses offer access to the profiling and other functionality in to `Tracer`, while adding custom method decorators. The custom decorators perform tasks such as writing the current basis to disk on every nth `Spikenet` iteration.
