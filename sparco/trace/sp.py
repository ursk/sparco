import sparco.sp
import trace.util
import trace.writer

profile_methods = [
    'infer_coefficients',
    'learn_basis1',
    'learn_basis2'
    ]
for f in profile_methods:
  trace.util.wrap(trace.util.time_track, f, sparco.sp.RootSpikenet)

# algorithm tracking

sparco.sp.RootSpikenet.bases += (trace.writer.Writer,)

def __init__(orig):
  def wrapped(*args, **kwargs):
    orig(*args, **kwargs)
    Writer.__init__(*args, **kwargs)
  return wrapped

trace.util.wrap(__init__, '__init__', sparco.sp.RootSpikenet)

def iteration(orig):
  def wrapped(*args, **kwargs):
    orig(*args, **kwargs)
    self = args[0]
    if self.t > 0 and self.write_interval and self.t % self.write_interval == 0:
      self.write_snapshot()
  return wrapped

trace.util.wrap(iteration, 'iteration', sparco.sp.RootSpikenet)

