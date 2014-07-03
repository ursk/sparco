import trace.util
import sparco.db

def patch_test(orig):
  def wrapped(*args, **kwargs):
    res = orig(*args, **kwargs)
    if not res:
      logging.info("patch failed test {0}".format(orig.__name__))
    return res
  return wrapped

sparco.db.DB.patch_filters = map(patch_test, sparco.DB.patch_filters)
