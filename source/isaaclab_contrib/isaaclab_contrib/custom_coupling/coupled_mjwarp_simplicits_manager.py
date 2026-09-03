# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MJWarp/Simplicits coupling manager used by the Gaussian-twin benchmark."""

from __future__ import annotations

import logging
from typing import Any

import torch
import warp as wp
from isaaclab.physics import PhysicsManager
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.timer import Timer
from isaaclab.utils.string import resolve_matching_names
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase, SolverMuJoCo

from .newton_manager_cfg import CoupledMJWarpSimplicitsSolverCfg
from .simplicits_assembly import build_gaussian_twin_simplicits_model

logger = logging.getLogger(__name__)


class _PrebuiltModelBuilder:
    """Narrow lifecycle adapter for an already-finalized Simplicits model.

    ``NewtonManager.start_simulation()`` normally owns a mutable
    :class:`newton.ModelBuilder`.  Simplicits must finalize that builder early
    to assemble its reduced coordinates, so only the three operations the
    remaining manager lifecycle actually needs are exposed here.  In
    particular, this is deliberately *not* a transparent proxy for the old
    builder: mutating it after finalization would not affect the model.
    """

    def __init__(self, model: Any):
        self._model = model
        self._up_axis = None

    @property
    def up_axis(self):
        return self._up_axis

    @up_axis.setter
    def up_axis(self, value) -> None:
        # The model is already finalized.  Retain the value solely because
        # NewtonManager sets it as part of its generic lifecycle.
        self._up_axis = value

    @property
    def shape_label(self):
        """Read-only finalized labels used by frame-transformer setup."""
        return self._model.shape_label

    def finalize(self, device: str, **_kwargs: Any) -> Any:
        del device
        return self._model

    def request_state_attributes(self, *attributes: str) -> None:
        self._model.request_state_attributes(*attributes)


class NewtonCoupledMJWarpSimplicitsManager(NewtonManager):
    """One-way MJWarp rigid simulation with Kaolin RKPM Simplicits objects.

    The one-way form is intentional for the benchmark: it gives the same
    robot/contact topology as the VBD/MJWarp baseline while avoiding a second
    hand-written reaction law that would make the comparison ambiguous.
    """

    _rigid_solver: SolverMuJoCo | None = None
    _simplicits_solver: Any = None
    _object_ids: list[list[int]] | None = None
    _z_indices: list[list[torch.Tensor]] | None = None
    _model_init_complete: bool = False

    @classmethod
    def start_simulation(cls) -> None:
        cfg = PhysicsManager._cfg
        if cfg is None or not isinstance(cfg.solver_cfg, CoupledMJWarpSimplicitsSolverCfg):
            raise RuntimeError("NewtonCoupledMJWarpSimplicitsManager requires CoupledMJWarpSimplicitsSolverCfg.")
        solver_cfg = cfg.solver_cfg
        if not solver_cfg.asset_paths:
            raise ValueError("Simplicits simulation needs at least one packaged RKPM asset.")

        stage = get_current_stage()
        env_paths, env_positions = cls._discover_environments(stage)
        if not env_paths:
            raise RuntimeError("No environment roots found below /World/envs while building Simplicits.")
        NewtonManager._num_envs = len(env_paths)
        with Timer(name="gaussian_twin_simplicits_build", msg="Simplicits model build took:"):
            model, object_ids, _particle_ranges = build_gaussian_twin_simplicits_model(
                stage=stage,
                env_paths=env_paths,
                env_positions=env_positions,
                asset_paths=solver_cfg.asset_paths,
                slot_positions=solver_cfg.slot_positions,
                device=PhysicsManager._device,
                up_axis=cls._up_axis,
                gravity=abs(float(cls._gravity_vector[2])) if cls._gravity_vector[2] else 9.81,
                max_quadrature_points=solver_cfg.max_quadrature_points,
                num_newton_steps=solver_cfg.num_newton_steps,
                cg_iterations=solver_cfg.cg_iterations,
                cg_tolerance=solver_cfg.cg_tolerance,
                newton_convergence_tolerance=solver_cfg.newton_convergence_tolerance,
                particle_radius=solver_cfg.collision_particle_radius,
                soft_contact_ke=solver_cfg.soft_contact_ke,
                soft_contact_mu=solver_cfg.soft_contact_mu,
                soft_contact_coefficient=solver_cfg.soft_contact_coefficient,
                enable_inter_object_collisions=solver_cfg.enable_inter_object_collisions,
                pre_finalize_callback=cls._configure_prebuilt_builder,
            )
        cls._object_ids = object_ids
        cls._z_indices = cls._object_z_indices(model, object_ids)
        NewtonManager._builder = _PrebuiltModelBuilder(model)
        try:
            super().start_simulation()
        except Exception:
            logger.exception("Simplicits model hand-off to NewtonManager failed")
            raise

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledMJWarpSimplicitsSolverCfg) -> None:
        if solver_cfg.rigid_solver_cfg.use_mujoco_contacts is not True:
            raise ValueError("Simplicits/MJWarp requires MJWarp's rigid contact solver.")
        try:
            from kaolin.experimental.newton.solver import SimplicitsSolver
        except ImportError as error:
            raise ImportError(
                "Simplicits/MJWarp requires Kaolin. Install the local Kaolin checkout into this Isaac Lab venv."
            ) from error
        cls._rigid_solver = solver_cfg.rigid_solver_cfg.class_type._create_solver(model, solver_cfg.rigid_solver_cfg)
        cls._simplicits_solver = SimplicitsSolver(model)
        NewtonManager._solver = SolverBase(model)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_contact_sensors = False
        NewtonManager._supports_rigid_body_force_input = True

    @classmethod
    def _configure_prebuilt_builder(cls, builder: Any) -> None:
        """Run normal pre-finalize callbacks while the real builder is mutable."""
        if cls._model_init_complete:
            raise RuntimeError("Simplicits MODEL_INIT was dispatched more than once during model assembly.")
        NewtonManager._builder = builder
        # The model is finalized manually immediately after this callback.
        # Dispatch MODEL_INIT and inject sites exactly here, while ``builder``
        # can still be changed.  The generic lifecycle below is then allowed
        # to initialize states around a narrow prebuilt-model adapter.
        super().dispatch_event(PhysicsEvent.MODEL_INIT)
        cls._cl_inject_sites_fallback()
        cls._model_init_complete = True

    @classmethod
    def _register_builder_attributes(cls, builder: Any) -> None:
        """Do not register builder arrays after the Simplicits model is finalized."""
        if isinstance(builder, _PrebuiltModelBuilder):
            return
        super()._register_builder_attributes(builder)

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: Any) -> None:
        """Skip mutable-builder normalization for the prebuilt-model adapter."""
        if isinstance(builder, _PrebuiltModelBuilder):
            return
        super()._prepare_builder_for_finalize(builder)

    @classmethod
    def _cl_inject_sites_fallback(cls) -> None:
        """Inject sites into the real multi-world builder exactly once.

        Newton's stock fallback assumes a single world and puts every matched
        body into one site-map row.  The Simplicits assembly constructs a flat
        multi-world builder manually, so partition matches by ``body_world``
        before the model is finalized.
        """
        if cls._model_init_complete:
            return
        builder = NewtonManager._builder
        body_labels = list(builder.body_label)
        for (body_pattern, per_world, _xform_key), (label, xform) in cls._cl_pending_sites.items():
            if body_pattern is None and not per_world:
                site_index = builder.add_site(body=-1, xform=xform, label=label)
                NewtonManager._cl_site_index_map[label] = (site_index, None)
                continue
            if per_world:
                raise NotImplementedError(
                    "Per-world root sites are not supported by the experimental Simplicits assembly; "
                    "attach the sensor to a rigid body instead."
                )
            matched_indices, _matched_names = resolve_matching_names(
                body_pattern, body_labels, raise_when_no_match=False
            )
            if not matched_indices:
                raise ValueError(
                    f"Site '{label}' with body pattern '{body_pattern}' matched no bodies in the Simplicits builder."
                )
            per_world_indices: list[list[int]] = [[] for _ in range(cls._num_envs)]
            for body_index in matched_indices:
                world_index = int(builder.body_world[body_index])
                if not 0 <= world_index < cls._num_envs:
                    raise ValueError(
                        f"Site '{label}' matched non-local body '{builder.body_label[body_index]}' "
                        f"in world {world_index}."
                    )
                site_index = builder.add_site(
                    body=body_index,
                    xform=xform,
                    label=f"{builder.body_label[body_index]}/{label}",
                )
                per_world_indices[world_index].append(site_index)
            NewtonManager._cl_site_index_map[label] = (None, per_world_indices)
        NewtonManager._cl_pending_sites.clear()

    @classmethod
    def dispatch_event(cls, event: PhysicsEvent, payload: Any = None) -> None:
        """Suppress only the duplicate generic ``MODEL_INIT`` dispatch."""
        if event is PhysicsEvent.MODEL_INIT and cls._model_init_complete:
            return
        super().dispatch_event(event, payload)

    @classmethod
    def _capture_or_defer_graph(cls) -> None:
        # Kaolin's reduced Newton solve uses Torch operations.  Capturing it in
        # Warp's graph is unsafe; a future Kaolin graph-safe adapter can replace
        # this conservative opt-out without changing task configuration.
        if PhysicsManager._cfg is not None and PhysicsManager._cfg.use_cuda_graph:
            logger.warning("Disabling CUDA graph capture for experimental Simplicits/MJWarp simulation.")
        NewtonManager._graph = None
        NewtonManager._graph_capture_pending = False

    @classmethod
    def step(cls) -> None:
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return
        if cls._model_changes:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    if cls._rigid_solver is not None:
                        cls._rigid_solver.notify_model_changed(change)
                    if cls._simplicits_solver is not None:
                        cls._simplicits_solver.notify_model_changed(change)
            NewtonManager._model_changes = set()
        super().step()

    @classmethod
    def _step_solver(
        cls, state_in: State, state_out: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Advance the robot, then the RKPM object against that new robot pose."""
        # Do not clear state_in: manager force callbacks (mouse picking,
        # external wrenches) run immediately before this function and MJWarp
        # is expected to consume them.
        state_out.clear_forces()
        # SolverMuJoCo operates exclusively on its compiled rigid MJWarp
        # model.  It neither reads nor integrates Newton particle arrays, so
        # do not mutate the shared model topology to hide Simplicits particles
        # around this call.
        cls._rigid_solver.step(state_in, state_out, control, None, substep_dt)
        # Kaolin's contact energy evaluates its input state.  Give it the new
        # MJWarp pose while retaining state_out as the final double buffer.
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)
        cls._simplicits_solver.step(state_in, state_out, control, contacts, substep_dt)

    @classmethod
    def _run_solver_substeps(cls, contacts: Contacts | None) -> None:
        collide = cls._needs_collision_pipeline and cls._collision_pipeline is not None and contacts is not None
        for substep in range(cls._num_substeps):
            for callback in cls._state_force_callbacks:
                callback(cls._state_0)
            if collide and (substep == 0 or (cls._collision_decimation > 0 and substep % cls._collision_decimation == 0)):
                cls._collision_pipeline.collide(cls._state_0, contacts)
            cls._step_solver(cls._state_0, cls._state_1, cls._control, contacts, cls._solver_dt)
            NewtonManager._state_0, NewtonManager._state_1 = cls._state_1, cls._state_0
            cls._state_0.clear_forces()

    @classmethod
    def _simulate_full(cls) -> None:
        contacts = cls._contacts if cls._needs_collision_pipeline else None
        physics_dt = cls._solver_dt * cls._num_substeps
        for _ in range(cls._decimation):
            if cls._adapter is not None:
                cls._adapter.step(cls._state_0, cls._control, physics_dt)
            for callback in cls._post_actuator_callbacks:
                callback()
            cls._run_solver_substeps(contacts)
        for callback in cls._post_step_callbacks:
            callback()
        cls._update_sensors(contacts)

    @classmethod
    def _simulate_physics_only(cls) -> None:
        contacts = cls._contacts if cls._needs_collision_pipeline else None
        cls._run_solver_substeps(contacts)
        for callback in cls._post_step_callbacks:
            callback()
        cls._update_sensors(contacts)

    @classmethod
    def get_simplicits_scene_and_object_ids(cls) -> tuple[Any, list[list[int]]]:
        if cls._model is None or cls._object_ids is None:
            raise RuntimeError("Simplicits model has not been initialized.")
        return cls._model.simplicits_scene, cls._object_ids

    @classmethod
    def reset_gaussian_twin_slot(cls, slot: int, env_ids, active: bool | torch.Tensor) -> None:
        """Restore one slot's reduced state, parking inactive slots below the table."""
        if cls._model is None or cls._object_ids is None or cls._z_indices is None or cls._state_0 is None:
            return
        from kaolin.physics.simplicits import simulation as simplicits_simulation

        ids = [int(value) for value in torch.as_tensor(env_ids, device="cpu").reshape(-1).tolist()]
        if not ids:
            return
        if isinstance(active, bool):
            active_by_env = [active] * len(ids)
        else:
            active_by_env = [bool(value) for value in torch.as_tensor(active, device="cpu").reshape(-1).tolist()]
            if len(active_by_env) != len(ids):
                raise ValueError("Simplicits reset active mask must have one entry per environment ID.")
        scene = cls._model.simplicits_scene
        sim_z = wp.to_torch(cls._state_0.sim_z).clone()
        sim_z_dot = wp.to_torch(cls._state_0.sim_z_dot).clone()
        for env_id, is_active in zip(ids, active_by_env, strict=True):
            obj = scene.sim_obj_dict[cls._object_ids[env_id][slot]]
            transform = cls._model._simplicits_initial_transforms[env_id][slot].clone()
            if not is_active:
                transform[2, 3] -= 100.0
            obj.init_transform = simplicits_simulation.torch_utilities.standard_transform_to_relative(transform)
            obj.reset_sim_state()
            indices = cls._z_indices[env_id][slot]
            sim_z.index_copy_(0, indices, obj.z.flatten())
            sim_z_dot.index_fill_(0, indices, 0.0)
        z = wp.from_torch(sim_z.contiguous())
        z_dot = wp.from_torch(sim_z_dot.contiguous())
        for state in (cls._state_0, cls._state_1):
            wp.copy(state.sim_z, z)
            wp.copy(state.sim_z_dot, z_dot)
        wp.copy(scene.sim_z, z)
        wp.copy(scene.sim_z_dot, z_dot)
        # Collision is queried before the next reduced solve. Reconstruct the
        # Newton particle slice immediately so the first post-reset contact
        # query observes the new (or parked) object pose rather than last
        # episode's particles.
        particle_q = cls._model.sim_z_to_full(z)
        particle_qd = cls._model.sim_z_dot_to_full(z_dot)
        count = cls._model.simplicits_particle_end - cls._model.simplicits_particle_start
        for state in (cls._state_0, cls._state_1):
            wp.copy(state.particle_q, particle_q, dest_offset=cls._model.simplicits_particle_start, count=count)
            wp.copy(state.particle_qd, particle_qd, dest_offset=cls._model.simplicits_particle_start, count=count)

    @staticmethod
    def _discover_environments(stage: Any) -> tuple[list[str], list[tuple[float, float, float]]]:
        root = stage.GetPrimAtPath("/World/envs")
        indexed: list[tuple[int, str, tuple[float, float, float]]] = []
        for child in root.GetChildren() if root and root.IsValid() else ():
            if not child.GetName().startswith("env_"):
                continue
            try:
                index = int(child.GetName().split("_", 1)[1])
            except ValueError:
                continue
            value = child.GetAttribute("xformOp:translate").Get()
            position = (float(value[0]), float(value[1]), float(value[2])) if value is not None else (0.0, 0.0, 0.0)
            indexed.append((index, child.GetPath().pathString, position))
        indexed.sort()
        return [item[1] for item in indexed], [item[2] for item in indexed]

    @staticmethod
    def _object_z_indices(model: Any, object_ids: list[list[int]]) -> list[list[torch.Tensor]]:
        """Return exact reduced-coordinate indices for every env/slot object.

        Kaolin currently allocates object blocks contiguously, but reset must
        not rely on that implementation detail: future packing and sparse
        reduced-coordinate layouts are free to reorder these maps.
        """
        indices_by_env: list[list[torch.Tensor]] = []
        for env_objects in object_ids:
            env_indices: list[torch.Tensor] = []
            for object_id in env_objects:
                mapping = wp.to_torch(model.simplicits_scene.object_to_z_map[object_id])
                env_indices.append(mapping.to(dtype=torch.long).clone())
            indices_by_env.append(env_indices)
        return indices_by_env

    @classmethod
    def _solver_specific_clear(cls) -> None:
        super()._solver_specific_clear()
        cls._rigid_solver = None
        cls._simplicits_solver = None
        cls._object_ids = None
        cls._z_indices = None
        cls._model_init_complete = False
