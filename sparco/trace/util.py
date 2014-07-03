import logging

def wrap(wrapper, method, klass):
  setattr(klass, method, wrapper(getattr(klass, method)))

PROFILING_TABLE = {}
def time_track(orig):
  PROFILING_TABLE[orig.__name__] = []
  def tracked_function(*args, **kwargs):
    start = time.time()
    res = orig(*args, **kwargs)
    end = time.time()
    PROFILING_TABLE[orig.__name__].append(end - start)
    return res
  return tracked_function if mpi.rank == mpi.root else orig

