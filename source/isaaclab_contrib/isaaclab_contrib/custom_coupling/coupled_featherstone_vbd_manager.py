# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic Featherstone and VBD coupling manager.

This is deliberately a one-way, kinematic coupling: it is the robust
DexSuite training topology, not a replacement for the native proxy coupler
used by the Gaussian-twin demonstration by default.
"""

from __future__ import annotations

import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase, SolverFeatherstone, SolverVBD

from isaaclab.physics import PhysicsManager

from .newton_manager_cfg import CoupledFeatherstoneVBDSolverCfg


@wp.kernel
def _position_targets_to_velocities(
    joint_q: wp.array(dtype=float),
    joint_target_q: wp.array(dtype=float),
    joint_velocity_limit: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_target_q_start: wp.array(dtype=int),
    joint_target_mode: wp.array(dtype=int),
    inv_dt: float,
    velocity_limit_scale: float,
    joint_qd: wp.array(dtype=float),
):
    """Convert actuated, one-coordinate-per-DoF targets to bounded velocities.

    Free and ball joints have a different number of position coordinates and
    velocity DoFs.  They cannot be driven by this simple finite difference, so
    leave them to Featherstone's ordinary dynamics rather than indexing a
    coordinate-layout target with a DoF index.
    """
    joint_index = wp.tid()
    q_start = joint_q_start[joint_index]
    qd_start = joint_qd_start[joint_index]
    coord_count = joint_q_start[joint_index + 1] - q_start
    dof_count = joint_qd_start[joint_index + 1] - qd_start
    if coord_count != dof_count:
        return

    target_start = joint_target_q_start[joint_index]
    for dof_offset in range(dof_count):
        dof_index = qd_start + dof_offset
        target_mode = joint_target_mode[dof_index]
        # JointTargetMode.POSITION and POSITION_VELOCITY, respectively.
        if target_mode == 1 or target_mode == 3:
            velocity = (joint_target_q[target_start + dof_offset] - joint_q[q_start + dof_offset]) * inv_dt
            limit = wp.abs(joint_velocity_limit[dof_index]) * velocity_limit_scale
            if limit > 0.0:
                velocity = wp.clamp(velocity, -limit, limit)
            joint_qd[dof_index] = velocity


class NewtonCoupledFeatherstoneVBDManager(NewtonVBDManager):
    """Advance a Featherstone robot kinematically, then solve VBD against it."""

    _rigid_solver: SolverFeatherstone | None = None
    _soft_solver: SolverVBD | None = None
    _velocity_limit: wp.array | None = None
    _gravity_zero: wp.array | None = None
    _gravity_saved: wp.array | None = None
    _joint_target_ke_zero: wp.array | None = None
    _joint_target_kd_zero: wp.array | None = None
    _joint_target_ke_saved: wp.array | None = None
    _joint_target_kd_saved: wp.array | None = None
    _velocity_limit_scale: float = 1.0

    @classmethod
    def step(cls) -> None:
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return
        if cls._model_changes:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    cls._rigid_solver.notify_model_changed(change)
                    cls._soft_solver.notify_model_changed(change)
                NewtonManager._model_changes = set()
        super().step()

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledFeatherstoneVBDSolverCfg) -> None:
        if not solver_cfg.soft_solver_cfg.integrate_with_external_rigid_solver:
            raise ValueError("Kinematic Featherstone/VBD coupling requires integrate_with_external_rigid_solver=True.")
        if NewtonManager._report_contacts:
            raise NotImplementedError(
                "Newton contact sensors are not supported by kinematic Featherstone/VBD coupling."
            )

        cls._rigid_solver = SolverFeatherstone(
            model, **cls._filter_solver_kwargs(SolverFeatherstone, solver_cfg.rigid_solver_cfg)
        )
        cls._soft_solver = SolverVBD(model, **cls._filter_solver_kwargs(SolverVBD, solver_cfg.soft_solver_cfg))
        cls._velocity_limit = model.joint_velocity_limit
        if cls._velocity_limit is None:
            cls._velocity_limit = wp.zeros(model.joint_dof_count, dtype=float, device=model.device)
        cls._gravity_zero = wp.zeros_like(model.gravity)
        cls._gravity_saved = wp.clone(model.gravity)
        cls._joint_target_ke_zero = wp.zeros_like(model.joint_target_ke)
        cls._joint_target_kd_zero = wp.zeros_like(model.joint_target_kd)
        cls._joint_target_ke_saved = wp.clone(model.joint_target_ke)
        cls._joint_target_kd_saved = wp.clone(model.joint_target_kd)
        cls._velocity_limit_scale = solver_cfg.kinematic_velocity_limit_scale
        if solver_cfg.shape_material_ke is not None:
            model.shape_material_ke.fill_(float(solver_cfg.shape_material_ke))
        if solver_cfg.shape_material_kd is not None:
            model.shape_material_kd.fill_(float(solver_cfg.shape_material_kd))
        if solver_cfg.shape_material_mu is not None:
            model.shape_material_mu.fill_(float(solver_cfg.shape_material_mu))

        NewtonManager._solver = SolverBase(model)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_contact_sensors = False
        NewtonManager._supports_rigid_body_force_input = True

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        if world_mask is None:
            return
        cls._rigid_solver.reset(cls._state_0, world_mask=world_mask, flags=0)
        cls._soft_solver.reset(cls._state_0, world_mask=world_mask, flags=0)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        super()._solver_specific_clear()
        cls._rigid_solver = None
        cls._soft_solver = None
        cls._velocity_limit = None
        cls._gravity_zero = None
        cls._gravity_saved = None
        cls._joint_target_ke_zero = None
        cls._joint_target_kd_zero = None
        cls._joint_target_ke_saved = None
        cls._joint_target_kd_saved = None
        cls._velocity_limit_scale = 1.0

    @classmethod
    def _simulate_physics_only(cls) -> None:
        if cls._model.particle_count > 0 and hasattr(cls._soft_solver, "rebuild_bvh"):
            cls._soft_solver.rebuild_bvh(cls._state_0)
        super()._simulate_physics_only()

    @classmethod
    def _step_solver(
        cls, state_in: State, state_out: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Advance the robot kinematically, then solve VBD against its new pose."""
        model = cls._model
        # ``state_in`` may contain forces from NewtonManager's framework
        # callbacks (for example, viewer drag picking).  The base manager
        # consumes and clears those after this substep; clearing only the
        # output buffer here prevents stale forces without discarding input.
        state_out.clear_forces()

        saved_particle_count = model.particle_count
        saved_shape_contact_pair_count = model.shape_contact_pair_count
        model.particle_count = 0
        model.gravity.assign(cls._gravity_zero)
        model.shape_contact_pair_count = 0
        model.joint_target_ke.assign(cls._joint_target_ke_zero)
        model.joint_target_kd.assign(cls._joint_target_kd_zero)
        try:
            wp.launch(
                _position_targets_to_velocities,
                dim=model.joint_type.shape[0],
                inputs=[
                    state_in.joint_q,
                    control.joint_target_q,
                    cls._velocity_limit,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_target_q_start,
                    model.joint_target_mode,
                    1.0 / substep_dt,
                    cls._velocity_limit_scale,
                ],
                outputs=[state_in.joint_qd],
                device=model.device,
            )
            cls._rigid_solver.step(state_in, state_out, control, None, substep_dt)
        finally:
            model.particle_count = saved_particle_count
            model.gravity.assign(cls._gravity_saved)
            model.shape_contact_pair_count = saved_shape_contact_pair_count
            model.joint_target_ke.assign(cls._joint_target_ke_saved)
            model.joint_target_kd.assign(cls._joint_target_kd_saved)
        # The base Newton manager owns collision scheduling and passes the
        # current contacts in.  Re-colliding here bypasses collision decimation
        # and needlessly performs the first substep twice.
        cls._soft_solver.step(state_in, state_out, control, contacts, substep_dt)
