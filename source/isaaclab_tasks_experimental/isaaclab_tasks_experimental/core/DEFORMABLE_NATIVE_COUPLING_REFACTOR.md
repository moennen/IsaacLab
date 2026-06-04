# Deformable Native Coupling Refactor

This branch was rebased onto `IsaacLab/develop` at `7e54511a6a49`. The sibling
`newton` worktree was updated to the IsaacLab-pinned `v1.2.0` tag from
`source/isaaclab_newton/pyproject.toml`.

## Review Findings

The previous manipulation attempt added task-local Newton managers, direct VBD
coupling kernels, and object adapters that exposed deformable particles through
rigid-object-like data fields. That approach is obsolete on current IsaacLab:

- `isaaclab_contrib.deformable.newton_manager_cfg` now provides
  `VBDSolverCfg`, `CoupledMJWarpVBDSolverCfg`, `CoupledFeatherstoneVBDSolverCfg`,
  and `NewtonModelCfg`.
- `NewtonCoupledMJWarpVBDManager` implements same-substep two-way coupling by
  applying body-particle reactions before the rigid MJWarp step and then
  advancing VBD for the soft object.
- `isaaclab_contrib.deformable.DeformableObject` is the native Newton-backed
  asset for volume and surface deformables.
- `Isaac-Lift-Soft-Franka-v0` and `Isaac-Lift-Cloth-Franka-v0` are trainable
  examples of a rigid manipulator interacting with deformable objects through
  the native asset and solver stack.

Because of that, the cleanup removes the old `franka_cube_pick`,
`franka_vbd_cube_pick`, `dexsuite_3dg`, and `dexsuite_3dg_proxy` prototype trees
instead of trying to forward-port their custom physics managers.

## Recommended Restart

Start from `source/isaaclab_tasks/isaaclab_tasks/core/lift_franka_soft/` rather
than the deleted prototype tasks.

1. Copy only the Kuka/Allegro scene, action, camera, and agent settings that are
   still useful. Do not copy custom Newton manager classes or monkey patches.
2. Represent the object as `isaaclab.assets.deformable_object.DeformableObjectCfg`
   in the scene, preferably under the asset name `deformable`.
3. For the first trainable version, use a small generated `MeshCuboidCfg` or a
   simple tetrahedral mesh with `NewtonDeformableBodyPropertiesCfg` and
   `NewtonDeformableBodyMaterialCfg`. Bring the Dexsuite mesh asset back only
   after reset, observation, and reward flow are stable.
4. Configure Newton with `CoupledMJWarpVBDSolverCfg(coupling_mode="two_way")`,
   `MJWarpSolverCfg` for the rigid robot, `VBDSolverCfg` for the deformable, and
   `NewtonModelCfg` for shared soft/rigid contact stiffness, damping, and
   friction.
5. Use observations and rewards based on native deformable data:
   `deformable.data.root_pos_w`, `deformable.data.root_vel_w`,
   `deformable.data.nodal_pos_w`, and sampled nodal positions in the robot root
   frame.
6. Keep the first task objective narrow: lift or stabilize a small deformable
   with Kuka/Allegro. Add full Dexsuite ragdoll geometry, Gaussian visual assets,
   and richer grasp validation only after a minimal trainable system is working.

The implementation should not subclass `NewtonManager`, write directly into
`state.body_f` from task code, or expose a fake rigid-object adapter for the
deformable. Those responsibilities now belong to the native IsaacLab/Newton
deformable stack.
