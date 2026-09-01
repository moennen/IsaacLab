# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MJWarp and VBD coupling manager."""

from __future__ import annotations

import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase, SolverMuJoCo, SolverVBD

from isaaclab.physics import PhysicsManager

from .kernels import _kernel_body_particle_reaction
from .newton_manager_cfg import CoupledMJWarpVBDSolverCfg


@wp.kernel
def _copy_particle_history_for_reset(
    source: wp.array(dtype=wp.vec3),
    destination: wp.array(dtype=wp.vec3),
    particle_world: wp.array(dtype=wp.int32),
    world_mask: wp.array(dtype=wp.bool),
):
    particle = wp.tid()
    world = particle_world[particle]
    if world >= 0 and world_mask[world]:
        destination[particle] = source[particle]


@wp.kernel
def _copy_body_history_for_reset(
    source: wp.array(dtype=wp.transform),
    destination: wp.array(dtype=wp.transform),
    body_world: wp.array(dtype=wp.int32),
    world_mask: wp.array(dtype=wp.bool),
):
    body = wp.tid()
    world = body_world[body]
    if world >= 0 and world_mask[world]:
        destination[body] = source[body]


class NewtonCoupledMJWarpVBDManager(NewtonVBDManager):
    """:class:`NewtonVBDManager` specialization for custom MJWarp and VBD coupling.

    Reuses the VBD manager's deformable stage handling and adds a custom rigid-deformable coupling step.
    Newton's :class:`CollisionPipeline` provides deformable contacts.
    """

    _rigid_solver: SolverMuJoCo | None = None
    _soft_solver: SolverVBD | None = None
    _coupling_mode: str | None = None
    _body_q_prev: wp.array | None = None
    _builder_attribute_solvers = (SolverMuJoCo,)

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        # Notify both sub-solvers of model changes.
        if cls._model_changes:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    cls._rigid_solver.notify_model_changed(change)
                    cls._soft_solver.notify_model_changed(change)
                NewtonManager._model_changes = set()
        super().step()

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledMJWarpVBDSolverCfg) -> None:
        """Construct the coupled solvers and populate the base lifecycle slots.

        VBD uses Newton's collision pipeline and separate input/output states, so the lifecycle flags are fixed.
        """
        if solver_cfg.coupling_mode not in ("one_way", "two_way"):
            raise ValueError("coupling_mode must be 'one_way' or 'two_way'.")
        if not solver_cfg.rigid_solver_cfg.use_mujoco_contacts:
            raise ValueError("The custom coupling manager requires MJWarp internal contacts.")
        if not solver_cfg.soft_solver_cfg.integrate_with_external_rigid_solver:
            raise ValueError("The custom coupling manager requires VBD external rigid-body integration.")
        if NewtonManager._report_contacts:
            raise NotImplementedError("Newton contact sensors are not supported by the custom coupling manager.")

        cls._coupling_mode = solver_cfg.coupling_mode

        cls._rigid_solver = solver_cfg.rigid_solver_cfg.class_type._create_solver(model, solver_cfg.rigid_solver_cfg)
        cls._soft_solver = solver_cfg.soft_solver_cfg.class_type._create_solver(model, solver_cfg.soft_solver_cfg)
        # Keep coupling history locally instead of reaching into SolverVBD's
        # private implementation.  Isaac Lab supports more than one Newton
        # package revision, and older supported versions do not expose VBD's
        # internal proxy-history snapshot.
        cls._body_q_prev = wp.clone(model.body_q, device=model.device)

        # The base lifecycle needs a solver slot; substeps use the two solvers above.
        NewtonManager._solver = SolverBase(model)
        NewtonManager._use_single_state = False
        NewtonManager._supports_contact_sensors = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_rigid_body_force_input = True

    @classmethod
    def _step_solver(
        cls, state_in: State, state_out: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Run one coupled substep.

        Args:
            state_in: Current read/write state.
            state_out: Next state.
            control: Joint-level control inputs.
            contacts: Unused; the coupling helpers use the manager-owned contact buffer.
            substep_dt: Substep timestep [s].
        """
        if cls._coupling_mode == "one_way":
            cls._step_one_way(state_in, state_out, control, substep_dt)
        else:
            cls._step_two_way(state_in, state_out, control, substep_dt)

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        """Reset both sub-solvers."""
        if world_mask is None:
            return
        if cls._rigid_solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        cls._rigid_solver.reset(cls._state_0, world_mask=world_mask, flags=0)
        cls._soft_solver.reset(cls._state_0, world_mask=world_mask, flags=0)
        # SolverVBD retains particle positions separately for friction
        # velocity estimation. Seed that history at reset so a contact already
        # present at t=0 cannot be interpreted as a full-frame displacement
        # from the origin.
        wp.launch(
            _copy_particle_history_for_reset,
            dim=cls._state_0.particle_q.shape[0],
            inputs=[
                cls._state_0.particle_q,
                cls._soft_solver.particle_q_prev,
                cls._model.particle_world,
                world_mask,
            ],
            device=cls._model.device,
        )
        if cls._body_q_prev is not None:
            wp.launch(
                _copy_body_history_for_reset,
                dim=cls._state_0.body_q.shape[0],
                inputs=[cls._state_0.body_q, cls._body_q_prev, cls._model.body_world, world_mask],
                device=cls._model.device,
            )

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Clear custom coupling state."""
        super()._solver_specific_clear()
        cls._rigid_solver = None
        cls._soft_solver = None
        cls._coupling_mode = None
        cls._body_q_prev = None

    @classmethod
    def _simulate_physics_only(cls) -> None:
        # Rebuild the BVH before stepping solvers that require it, such as VBD cloth.
        if hasattr(cls._soft_solver, "rebuild_bvh"):
            cls._soft_solver.rebuild_bvh(cls._state_0)
        super()._simulate_physics_only()

    @classmethod
    def _step_one_way(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance rigid bodies and particles without deformable reaction forces."""
        # 1. Clear output forces.
        state_out.clear_forces()

        # 2. Detect deformable-rigid contacts.
        cls._collision_pipeline.collide(state_in, cls._contacts)

        # 3. Advance rigid bodies without injected deformable reactions.
        cls._rigid_step(state_in, state_out, control, dt)

        # 4. Advance particles using the updated rigid poses.
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _step_two_way(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance rigid bodies and particles with deformable reaction forces."""
        # 1. Clear output forces.
        state_out.clear_forces()

        # 2. Detect contacts before advancing rigid bodies.
        cls._collision_pipeline.collide(state_in, cls._contacts)

        # 3. Inject contact reactions before MJWarp consumes body_f. Cached
        # rigid and particle histories supply the friction velocity estimate.
        if state_in.body_f is not None:
            cls._apply_reactions(state_in, dt)

        # 4. Advance rigid bodies with the injected reactions.
        # Record the current pose after applying its reaction so the next
        # substep computes rigid contact-point velocity from two consecutive
        # input states.  ``state_out`` is an integration target and must not
        # be used as history before the rigid solver writes it.
        if cls._body_q_prev is not None:
            wp.copy(cls._body_q_prev, state_in.body_q)
        cls._rigid_step(state_in, state_out, control, dt)

        # 5. Advance particles using the contacts detected above.
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _rigid_step(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance rigid bodies with the configured sub-solver."""
        cls._rigid_solver.step(state_in, state_out, control, None, dt)

    @classmethod
    def _apply_reactions(cls, state: State, dt: float) -> None:
        """Inject normal and friction reaction forces into body_f.

        Args:
            state: Current particle and body state.
            dt: Substep timestep [s].
        """
        model = cls._model
        contacts = cls._contacts
        if contacts is None:
            return

        contact_capacity = int(contacts.soft_contact_particle.shape[0])
        if contact_capacity == 0:
            return

        # VBD mutates particle_q in place, so use its cached particle-position
        # history for the contact velocity estimate.
        wp.launch(
            _kernel_body_particle_reaction,
            dim=contact_capacity,
            inputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_shape,
                contacts.soft_contact_indices,
                contacts.soft_contact_barycentric,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                state.particle_q,
                cls._soft_solver.particle_q_prev,
                model.particle_radius,
                state.body_q,
                cls._body_q_prev,
                state.body_qd,
                model.body_com,
                model.shape_body,
                model.shape_margin,
                # This runs before SolverVBD initializes per-contact material
                # buffers for the freshly detected contacts.  Use the model
                # values here rather than stale (or first-step unallocated)
                # VBD buffers; the contact evaluator still handles particle,
                # edge, and face records identically to VBD.
                float(model.soft_contact_ke),
                float(model.soft_contact_kd),
                float(model.soft_contact_mu),
                float(cls._soft_solver.friction_epsilon),
                float(dt),
                state.body_f,
            ],
        )
