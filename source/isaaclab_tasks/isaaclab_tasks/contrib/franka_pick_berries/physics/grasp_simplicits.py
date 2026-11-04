# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional Kaolin Simplicits tissue with captured MLS contact and kinematic Franka pads."""

import inspect
import math
import re
from dataclasses import field

import newton
import numpy as np
import warp as wp
from isaaclab_newton.physics import NewtonSolver, NewtonSolverCfg
from newton._src.solvers.coupled.interface import CouplingInterface  # experimental; not yet exported by Newton

from isaaclab.utils import configclass


@wp.kernel(enable_backward=False)
def _body_transforms(
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3f),
    bodies: wp.array(dtype=int),
    time: float,
    output: wp.array(dtype=wp.mat44f),
):
    i = wp.tid()
    body = bodies[i]
    pose = body_q[body]
    linear = wp.spatial_top(body_qd[body])
    angular = wp.spatial_bottom(body_qd[body])
    orientation = wp.transform_get_rotation(pose)
    speed = wp.length(angular)
    if speed > 1e-8:
        orientation = wp.quat_from_axis_angle(angular / speed, speed * time) * orientation
    center = wp.transform_point(pose, body_com[body]) + time * linear
    translation = center - wp.quat_rotate(orientation, body_com[body])
    rotation = wp.quat_to_matrix(orientation)
    output[i] = wp.mat44f(
        rotation[0, 0],
        rotation[0, 1],
        rotation[0, 2],
        translation[0],
        rotation[1, 0],
        rotation[1, 1],
        rotation[1, 2],
        translation[1],
        rotation[2, 0],
        rotation[2, 1],
        rotation[2, 2],
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


@wp.kernel(enable_backward=False)
def _publish_particles(
    points: wp.array(dtype=wp.vec3f),
    start: int,
    dt: float,
    old: wp.array(dtype=wp.vec3f),
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    velocities[start + i] = (points[i] - old[start + i]) / dt
    positions[start + i] = points[i]


def _contact_data(physics, normals, areas, *, is_static=False):
    """Surface-only MLS: source and evaluation samples coincide, with explicit rest-area weights."""
    import torch

    data = {
        "X_band": physics.pts,
        "d_band": torch.zeros_like(physics.pts[:, 0]),
        "u_band": normals,
        "W_band": physics.skinning_weights,
        "dW_dX_band": physics.dwdx,
        "X_eval": physics.pts,
        "u_eval": normals,
        "w_eval": areas,
        "W_eval": physics.skinning_weights,
        "dW_dX_eval": physics.dwdx,
        "is_static": is_static,
    }

    if is_static:
        # Kinematic obstacles provide fields; query the moving tissue surface against
        # them once, without reverse obstacle queries into the deforming tissue field.
        for key in ("X_eval", "u_eval", "w_eval", "W_eval", "dW_dX_eval"):
            data[key] = data[key][:0]
    return data


class SolverSimplicits(newton.solvers.SolverBase, CouplingInterface):
    """Advance a baked Kaolin scene and publish its quadrature points into Newton state.

    Fixed contact capacity lets the outer Newton graph include Kaolin's captured MLS
    operations. The gripper drives kinematic contact surfaces; proxy feedback is explicitly zero.
    """

    def __init__(self, model, cfg):
        import torch
        from kaolin.physics.simplicits import SimplicitsScene, SkinnedPhysicsPoints

        super().__init__(model)
        self.cfg = cfg
        self._outer_pool = torch.cuda.graph_pool_handle()
        self.supports_outer_capture = {"mls_fixed_capacity", "async_solve_check"}.issubset(
            inspect.signature(SimplicitsScene).parameters
        ) and cfg.contact_capacity is not None
        capture_options = (
            {"mls_fixed_capacity": cfg.contact_capacity, "async_solve_check": True}
            if self.supports_outer_capture
            else {}
        )
        self.scene = SimplicitsScene(
            device=str(model.device),
            capturable=cfg.captured,
            **capture_options,
            timestep=cfg.timestep,
            max_newton_steps=cfg.newton_iterations,
            max_ls_steps=20,
            newton_hessian_regularizer=cfg.hessian_regularizer,
            conv_tol=1e-7,
        )
        self.object_ids = []
        self.particle_starts = []
        collision_data = {}
        start = 0
        for data, placement in zip(cfg.representations, cfg.placements, strict=True):
            fields = {
                key: torch.as_tensor(data[key], dtype=torch.float32, device=str(model.device))
                for key in ("pts", "yms", "prs", "rhos", "appx_vol", "skinning_weights", "dwdx")
            }
            physics = SkinnedPhysicsPoints(**fields)
            obj = self.scene.add_object(
                physics,
                init_transform=torch.as_tensor(placement, device=str(model.device)),
                apply_qr=False,
            )
            self.object_ids.append(obj)
            self.particle_starts.append(start)
            start += len(data["pts"])
            samples = data["collision"]

            def tensor(key):
                return torch.as_tensor(samples[key], dtype=torch.float32, device=str(model.device))

            contact_physics = SkinnedPhysicsPoints(
                tensor("pts"),
                torch.full_like(tensor("pts")[:, 0], float(data["yms"][0])),
                torch.full_like(tensor("pts")[:, 0], float(data["prs"][0])),
                torch.full_like(tensor("pts")[:, 0], float(data["rhos"][0])),
                fields["appx_vol"],
                tensor("weights"),
                tensor("dwdx"),
            )
            collision_data[obj] = _contact_data(contact_physics, tensor("normals"), tensor("areas"))
        if start != model.particle_count:
            raise ValueError("Simplicits quadrature counts must match the tissue particle view")
        self._finger_bodies = [
            i for i, label in enumerate(model.body_label) if re.search(r"panda_(left|right)finger", label)
        ]
        if len(self._finger_bodies) != 2:
            raise ValueError("The Simplicits tissue entry requires both Franka finger proxies")
        self._body_ids = wp.array(self._finger_bodies, dtype=int, device=model.device)
        self._poses = wp.empty(2, dtype=wp.mat44f, device=model.device)
        self._kinematic_ids = []
        from .finger_pads import PAD_HALF_EXTENTS, PAD_OFFSETS

        for body in self._finger_bodies:
            side = "left" if "leftfinger" in model.body_label[body] else "right"
            mesh = self._box_mesh(PAD_HALF_EXTENTS)
            mesh.apply_translation(PAD_OFFSETS[side])
            obj, contact = self._rigid_surface(mesh, cfg.contact_points)
            self._kinematic_ids.append(obj)
            collision_data[obj] = contact
        # Use the same static collision boxes authored for the other tissue solvers.
        import trimesh
        from scipy.spatial.transform import Rotation

        static_meshes = []
        scales, transforms = model.shape_scale.numpy(), model.shape_transform.numpy()
        for shape, label in enumerate(model.shape_label):
            if not label.startswith("/World/TablewareCollisions/"):
                continue
            if model.shape_source[shape] is not None:
                raise ValueError("Simplicits tableware expects the task's box collision proxies")
            mesh = self._box_mesh(scales[shape])
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(transforms[shape, 3:]).as_matrix()
            matrix[:3, 3] = transforms[shape, :3]
            mesh.apply_transform(matrix)
            static_meshes.append(mesh)
        if static_meshes:
            obj, contact = self._rigid_surface(trimesh.util.concatenate(static_meshes), cfg.contact_points * 4)
            collision_data[obj] = contact
        gravity = wp.to_torch(model.gravity)[0]
        # Kaolin stores the gradient of gravitational potential; Newton stores acceleration.
        self.scene.set_scene_gravity(-gravity)
        self.scene.set_scene_floor(
            floor_axis=2, floor_height=0, floor_penalty=cfg.floor_stiffness, friction=cfg.friction
        )
        self.scene.enable_collisions(
            mode="mls",
            mls_object_data=collision_data,
            mls_kernel_radius=cfg.kernel_radius,
            mls_barrier_extent=cfg.contact_margin,
            mls_barrier_mode="quadratic",
            collision_penalty=cfg.contact_stiffness,
            mls_mu=cfg.friction,
            mls_friction_fluid=1e-4,
        )

    @staticmethod
    def _box_mesh(half):
        import trimesh

        return trimesh.creation.box(extents=2 * np.asarray(half))

    def _rigid_surface(self, mesh, count):
        import torch
        from kaolin.physics.simplicits import PhysicsPoints, SimplicitsObject, sample_mesh_surface

        device = str(self.model.device)
        vertices = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        faces = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.long, device=device)
        points, normals = sample_mesh_surface(vertices, faces, count, device=device)
        physics = PhysicsPoints(points, 1e6, 0.3, 1000.0, max(float(abs(mesh.volume)), 1e-9))
        rigid = SimplicitsObject.create_rigid(physics_points=physics).bake(
            sampling_indices=torch.arange(len(points), device=device)
        )
        obj = self.scene.add_object(rigid, is_kinematic=True, apply_qr=False)
        areas = torch.full_like(points[:, 0], float(mesh.area) / len(points))
        return obj, _contact_data(rigid, normals, areas, is_static=True)

    def step(self, state_in, state_out, control, contacts, dt):
        import torch

        with torch.cuda.stream(wp.stream_to_torch(self.model.device)):
            capturing = self.model.device.is_capturing
            index = torch.device(wp.device_to_torch(self.model.device)).index or 0
            if capturing:
                torch._C._cuda_beginAllocateCurrentStreamToPool(index, self._outer_pool)
            try:
                self._step(state_in, state_out, dt)
            finally:
                if capturing:
                    torch._C._cuda_endAllocateToPool(index, self._outer_pool)

    def _step(self, state_in, state_out, dt):
        h = dt / self.cfg.substeps
        if not math.isclose(self.scene.timestep, h, rel_tol=1e-6):
            raise ValueError(
                "Simplicits timestep differs from its prepared scene; rebuild the solver after changing dt"
            )
        if state_out is not state_in:
            wp.copy(state_out.particle_q, state_in.particle_q)
        for substep in range(self.cfg.substeps):
            # Coupled proxy poses describe the tick's start. Match the explicit MPM pad
            # convention and advance at the substep midpoint using the supplied twist.
            wp.launch(
                _body_transforms,
                2,
                [
                    state_in.body_q,
                    state_in.body_qd,
                    self.model.body_com,
                    self._body_ids,
                    (substep + 0.5) * h,
                    self._poses,
                ],
                device=self.model.device,
            )
            poses = wp.to_torch(self._poses)
            for i, obj in enumerate(self._kinematic_ids):
                self.scene.set_kinematic_object_transform(obj, poses[i])
            self.scene.run_sim_step()
            for obj, start in zip(self.object_ids, self.particle_starts, strict=True):
                points = self.scene.get_object_deformed_pts(obj).contiguous()
                wp.launch(
                    _publish_particles,
                    len(points),
                    [
                        wp.from_torch(points, dtype=wp.vec3f),
                        start,
                        h,
                        state_out.particle_q,
                        state_out.particle_q,
                        state_out.particle_qd,
                    ],
                    device=self.model.device,
                )

    def reset(self, state, world_mask=None, flags=None):
        self.scene.reset_scene()

    def coupling_harvest_proxy_wrenches(self, body_local_to_proxy_global, out_body_f, **kwargs):
        """One-way kinematic contact contributes no reaction wrench to the shared accumulator."""
        return


class GraspSimplicitsSolver(NewtonSolver):
    """Construct the optional Kaolin adapter through the normal Newton solver configuration."""

    solver_class = SolverSimplicits
    single_state = True
    contact_sensors = False

    @classmethod
    def build(cls, cfg, model, mode, env_template=None):
        return SolverSimplicits(model, cfg)

    @classmethod
    def supports_capture(cls, cfg, solver):
        return cfg.captured and solver.supports_outer_capture

    @classmethod
    def uses_collision_pipeline(cls, cfg, solver):
        return False

    @classmethod
    def applies_body_forces(cls, cfg, solver):
        return False


@configclass
class GraspSimplicitsSolverCfg(NewtonSolverCfg):
    """Baked berry representations and MLS settings; importing this config does not import Kaolin."""

    class_type: type = GraspSimplicitsSolver
    representations: list = field(default_factory=list, repr=False)
    """Baked quadrature, material and MLS surface arrays, supplied by the asset loader."""
    placements: list = field(default_factory=list)
    """World transforms of the baked representations."""
    substeps: int = 4
    """Internal tissue substeps per coupled tick, with interpolated pad motion."""
    timestep: float = 1 / 480
    """Substep duration [s], set from the task simulation and coupling configuration before scene preparation."""
    captured: bool = True
    """Capture the Newton/MLS step and permit outer Newton physics capture with fixed contact capacity."""
    newton_iterations: int = 10
    """Maximum nonlinear iterations per physics step."""
    hessian_regularizer: float = 1e-4
    """Kaolin Hessian regularization for redundant learned handle modes."""
    contact_capacity: int = 32
    """Maximum MLS neighbors per surface sample; overflow raises on GPU and requires a larger capacity."""
    contact_points: int = 128
    """Surface samples per pad; static tableware uses four times this count."""
    kernel_radius: float = 0.006
    """MLS surface support radius [m]."""
    contact_margin: float = 0.0015
    """Quadratic contact barrier extent [m]."""
    contact_stiffness: float = 1e5
    """Kaolin MLS penalty coefficient, weighted by surface area."""
    floor_stiffness: float = 1e7
    """Kaolin floor penalty coefficient, weighted by integration volume."""
    friction: float = 1.0
    """Lagged Coulomb friction coefficient at pads, tableware and the floor."""
