Fixed
^^^^^

* Reduced native Gaussian-twin MJWarp/VBD cost and instability by using an interactive solver budget,
  particle contacts by default, and deformable-appropriate robot contact, friction, and gripper-force limits. Set
  ``ISAACLAB_GAUSSIAN_TWIN_FULL_SURFACE_CONTACT=1`` to restore the more expensive edge/face contact path.
