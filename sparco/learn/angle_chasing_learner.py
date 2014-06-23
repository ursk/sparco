class AngleChasingLearner(object):
  """
  Try to keep basis phi (treated as single vector) changing
  a specified number of degrees on each step.
  """

  def update_phi(self):
    if self.phi_angle < self.max_tolerable_angle:
      do_center = self.centering_interval and self.t % self.centering_interval == 0
      self.phi = center(self.proposed_phi) if do_center else self.proposed_phi
      self.phi = smooth(self.phi) if smooth else self.phi
    else:
      print 'Update to phi too large. Rejecting.'

  def update_eta(self):
    if angle < self.target:
      self.eta *= self.eta_up_factor
    else:
      self.eta *= self.eta_down_factor
