import trace.util
import sparco.sampler

class Tracer(traceutil.tracer.Tracer):

  # custom decorators

  def t_patch_test(tracer, orig, self, *args, **kwargs):
    res = orig(self, *args, **kwargs)
    if not res:
      logging.info("patch failed test {0}".format(orig.__name__))
    return res

sparco.sampler.Sampler.patch_filters = map(patch_test, sparco.Sampler.patch_filters)
