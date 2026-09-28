Added
^^^^^

* Added the full EBC Gaussian room and its corrected alignment to the USD-only
  task bundle. Made the cached EBC room the Gaussian-twin startup default and
  framed the robot workcell with a closer camera, without changing robot/table
  physics. Preserved custom background overrides and explicit disabling through
  ``ISAACLAB_DISABLE_ALIGNED_BACKGROUND=1`` or ``ISAACLAB_ENABLE_ALIGNED_BACKGROUND=0``.

* Added a task-local EBC bundle updater with manifest backup and a captured-frame
  option for startup checks. Existing bundles must be updated and republished;
  the previous Nova Carter/Galileo background remains available as an override.
