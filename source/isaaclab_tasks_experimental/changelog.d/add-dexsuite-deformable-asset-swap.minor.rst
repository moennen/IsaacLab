Added
^^^^^

* Added a play-only variant of the DexSuite deformable task that bakes all candidate
  assets in every environment and activates a random one on each reset (masking the
  rest via Newton ``particle_flags``), so the checkpoint can be replayed across the
  full asset set without impacting training.
