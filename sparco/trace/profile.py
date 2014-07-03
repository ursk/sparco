import json
import time

def time_track(orig, lst):
  def tracked_function(*args, **kwargs):
    start = time.time()
    res = orig(*args, **kwargs)
    end = time.time()
    lst.append(end - start)
    return res
  return tracked_function if mpi.rank == mpi.root else orig

class Profiler(object):

  def __init__(self, output_path=None, function_tree=None):
    self.table = {}
    self.output_path = output_path
    self.function_tree = function_tree
    self.wrap_functions(self.function_tree)

  def wrap_functions(self, tree, *keys):
    for k,v in tree.items():
      if isinstance(v, dict):
        self.table[k] = {}
        self.wrap_functions(v, *keys, k)
      elif k == 'funcs':
        parent = reduce(lambda a,k: getattr(a, k), keys[1:], keys[0])
        table_parent = reduce(lambda a,k: a[k], keys, self.table)
        for func in v:
          table_parent[func] = []
          setattr(parent, func, time_track(getattr(parent, func), table_parent[func]))

  def write_table(self):
    with open(self.output_path, 'w') as f:
      f.write(json.dumps(table, sort_keys=True, indent=4, separators=(',', ': ')))
