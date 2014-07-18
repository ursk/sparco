class AngleChasingLearner(object):
  """
  Try to keep basis phi (treated as single vector) changing
  a specified number of degrees on each step.
  """

  def update_phi(self):
    if self.phi_angle < self.max_angle:
      do_center = self.basis_centering_interval and self.t % self.basis_centering_interval == 0
      self.phi = center(self.proposed_phi) if do_center else self.proposed_phi
      return self.phi

  def update_eta(self):
    if self.phi_angle < self.target_angle:
      self.eta *= self.eta_up_factor
    else:
      self.eta *= self.eta_down_factor
