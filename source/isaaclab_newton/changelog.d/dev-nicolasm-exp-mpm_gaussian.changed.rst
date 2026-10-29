* Updated Newton to commit ``6542c2808ffbf317bf37bc3ca7bb13f5fc34d108`` and MuJoCo/MuJoCo Warp to 3.16 for
  Newton's implicit MPM grasping fixes, keeping workspace and wheel dependencies aligned.
* Changed :meth:`~isaaclab_newton.physics.MPMSolver.prepare_builder` to keep the mass and inertia of kinematic
  bodies. Newton's implicit MPM already treats kinematic bodies as infinite-mass colliders, and MuJoCo 3.16 rejects
  moving bodies without mass.
