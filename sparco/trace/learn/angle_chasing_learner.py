import logging
import traceutil.tracer

class Tracer(traceutil.tracer.Tracer):

  ########### CUSTOM DECORATORS

  def t_update_phi(tracer, orig, self, *args, **kwargs):
    if self.phi_angle >= self.max_angle:
      msg = 'Iteration {0}: Angle {1} > {2}. Update to phi too large. Rejecting.'
    else:
      msg = 'Iteration {0}: Angle {1} < {2}. Update OK.'
    logging.info(msg.format(self.t, self.phi_angle, self.max_angle))
    orig(self, *args, **kwargs)

  def t_update_eta(tracer, orig, self, *args, **kwargs):
    old_eta = self.eta
    orig(self, *args, **kwargs)

  wrappers = {
      'update_phi': [t_update_phi],
      'update_eta': [t_update_eta]
      }
