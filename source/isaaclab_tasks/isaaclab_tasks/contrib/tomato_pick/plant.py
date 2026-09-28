# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-workcell plant adapter; all contact and bending use Newton's MuJoCo solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from scipy.spatial.transform import Rotation

from pxr import Usd

from isaaclab.assets.asset_base import AssetBase
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.gaussian_tasks.usd_task_data import read_data


@configclass
class TomatoPlantCfg(DeformableObjectCfg):
    """Passive plant with calibrated bending and force-triggered fruit welds."""

    class_type: type = "{DIR}.plant:TomatoPlant"
    package_path: str = ""
    wind_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Ambient wind [m/s] in world coordinates."""
    bending_armature: float = 1.0e-5
    """Numerical rotational inertia [kg*m^2]; changes transient response, not static stiffness."""
    force_filter_tau: float = 0.03
    """Reaction-force smoothing time [s]."""
    separation_force_scale: float = 1.0
    """Multiplier on fruit breaking forces; positive, dimensionless."""
    separation_dwell: float = 0.03
    """Time above breaking force required before separation [s]."""


class TomatoPlant(AssetBase):
    """Synchronize a segmented plant and 16 independently releasable rigid fruits.

    Configurable stem armature defaults to 1e-5 kg*m^2 per bending DOF.
    This numerical regularization changes transient rotational inertia.
    Above-ground links remain passive; only the buried root and pot are fixed.
    """

    def __init__(self, cfg: TomatoPlantCfg):
        self.package = Path(cfg.package_path)
        stage = Usd.Stage.Open(str(self.package / "plant_visual.usdz"))
        root = str(stage.GetDefaultPrim().GetPath())
        self.manifest = read_data(stage, root + "/TaskData/Manifest")
        self.names = read_data(stage, root + "/TaskData/Bindings")
        self.calibration = read_data(stage, root + "/TaskData/Calibration")
        self._leaf_areas = np.asarray(read_data(stage, root + "/TaskData/LeafAreas"))
        self._mjcf = stage.GetDefaultPrim().GetAttribute("tomato:mjcf").Get()
        self._bindings = []
        self.filtered_force = np.zeros(16, dtype=np.float32)
        self.detached = np.zeros(16, dtype=bool)
        self._dwell = np.zeros(16)
        self.release_events = []
        self._elapsed = 0.0
        self.thresholds = np.array(
            [
                1.5 if f["ripeness"] == "red" else 2.5 if f["ripeness"] == "orange" else 4.0
                for f in self.manifest["fruits"]
            ]
        )
        if not np.isfinite(cfg.separation_force_scale) or cfg.separation_force_scale <= 0:
            raise ValueError("separation_force_scale must be finite and positive")
        self.thresholds *= cfg.separation_force_scale
        super().__init__(cfg)
        NewtonManager._per_world_builder_hooks.append(self._build)

    @property
    def num_instances(self) -> int:
        return 1

    @property
    def data(self):
        return self

    def _build(self, builder, index, position, quaternion):
        import newton

        if index != 0:
            raise ValueError("Tomato picking currently supports --num_envs 1 only.")
        self._origin = np.asarray(self.cfg.init_state.pos) + np.asarray(position)
        self._rotation = np.asarray(self.cfg.init_state.rot)
        pose = wp.transform(self._origin.tolist(), self._rotation.tolist())
        first_body, first_shape = builder.body_count, builder.shape_count
        builder.add_mjcf(self._mjcf, xform=pose, floating=None, collapse_fixed_joints=False)
        self._plant_body_ids = np.arange(first_body, builder.body_count)
        for j, body in enumerate(builder.joint_child):
            if body >= first_body and builder.joint_type[j] == newton.JointType.FREE:
                v = builder.joint_qd_start[j]
                builder.joint_armature[v : v + 6] = [1.0e-7] * 6
        self._physical_inertia = np.asarray(builder.body_inertia).reshape(-1, 3, 3).copy()
        lookup = {name.split("/")[-1]: i for i, name in enumerate(builder.body_label)}
        self.fruit_body_ids = np.array([lookup[f["id"]] for f in self.manifest["fruits"]])
        self._path_bodies = [np.array([lookup[n] for n in path]) for path in self.names["paths"]]
        # Imported MJCF colliders are invisible; source USD provides the textured surfaces.
        for i in range(first_shape, builder.shape_count):
            builder.shape_flags[i] &= ~int(newton.ShapeFlags.VISIBLE)
        objects = {o["id"]: o for o in self.manifest["objects"] if o["kind"] in ("fruit", "leaf")}
        self._stem_shapes = []
        for i in range(first_shape):
            label = str(builder.shape_label[i])
            if "/TomatoPlant/" not in label:
                continue
            for name, obj in objects.items():
                if "/" + name + "/" in label:
                    body = lookup[name]
                    builder.shape_body[i] = body
                    builder.shape_transform[i] = wp.transform_multiply(
                        wp.transform_inverse(builder.body_q[body]), builder.shape_transform[i]
                    )
                    break
            for obj in self.manifest["objects"]:
                if obj["kind"] == "stem" and "/" + obj["id"] + "/" in label:
                    self._stem_shapes.append((i, obj["path_index"]))
                    builder.shape_color[i] = (0.12, 0.22, 0.035)
        self._leaf_ids = np.array([lookup[x["id"]] for x in self.manifest["leaves"]])
        # Flat rubber-pad proxies on the inner fingertip faces. The stock convex
        # finger hull has a sloped tip that ejects these small spherical fruits.
        for name, side in (("panda_leftfinger", -1), ("panda_rightfinger", 1)):
            builder.add_shape_box(
                lookup[name],
                xform=wp.transform((0.0, side * 0.004, 0.040), wp.quat_identity()),
                hx=0.009,
                hy=0.004,
                hz=0.016,
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=1.0, gap=0.0),
                color=(0.035, 0.035, 0.035),
                label=name + "_tomato_rubber_pad",
            )

    def _initialize_impl(self):
        model = NewtonManager.get_model()
        # Restore authored SI inertias after the generic builder's small-inertia floor.
        inertias = model.body_inertia.numpy()
        inv = model.body_inv_inertia.numpy()
        for i in self._plant_body_ids:
            inertias[i] = self._physical_inertia[i]
            if np.linalg.det(inertias[i]) > 0:
                inv[i] = np.linalg.inv(inertias[i])
        model.body_inertia.assign(inertias)
        model.body_inv_inertia.assign(inv)
        self._rest_q = NewtonManager._state_0.body_q.numpy().copy()
        self._rest_joint_q = NewtonManager._state_0.joint_q.numpy().copy()
        self._bind_stems(model)
        colors = model.shape_color.numpy()
        colors[[shape for shape, path in self._stem_shapes]] = (0.12, 0.22, 0.035)
        model.shape_color.assign(colors)
        self._solver_ready = False
        self._step_dt = SimulationContext.instance().get_physics_dt()
        NewtonManager.register_post_step_callback(self._check_separation)
        NewtonManager.register_state_force_callback(self._apply_wind)

    def _configure_solver(self):
        solver = NewtonManager._solver
        m = solver.mj_model
        matched = 0
        self._calibrated_joints = []
        for j in range(m.njnt):
            name = m.joint(j).name
            param = next((v for n, v in self.calibration["joints"].items() if name.endswith("_" + n)), None)
            if param:
                q, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
                m.jnt_stiffness[j] = param["stiffness"]
                m.dof_armature[v : v + 3] = self.cfg.bending_armature
                m.dof_damping[v : v + 3] = param["damping"]
                m.qpos_spring[q : q + 4] = param["rest_quaternion_wxyz"]
                self._calibrated_joints.append((j, q, v, param))
                matched += 1
        if matched != len(self.calibration["joints"]):
            raise RuntimeError(f"Only {matched} of {len(self.calibration['joints'])} bending joints matched.")
        for j in range(m.njnt):
            if m.jnt_type[j] == 0 and "tomato_plant" in m.joint(j).name:
                v = m.jnt_dofadr[j]
                m.dof_armature[v : v + 6] = 1.0e-7
                m.dof_damping[v + 3 : v + 6] = 2.0e-6
        import mujoco

        mujoco.mj_setConst(m, mujoco.MjData(m))
        m.geom_gap[:] = 0
        # Gravity compensation is a robot actuator aid; plant gravity is fully active.
        for b in range(m.nbody):
            if "Robot" in m.body(b).name:
                m.body_gravcomp[b] = 1.0
        if solver.mjw_model is not None:
            for field in (
                "jnt_stiffness",
                "qpos_spring",
                "dof_damping",
                "dof_armature",
                "geom_gap",
                "body_gravcomp",
                "body_invweight0",
                "dof_invweight0",
                "actuator_acc0",
            ):
                arr = getattr(solver.mjw_model, field)
                arr.assign(np.broadcast_to(getattr(m, field), arr.numpy().shape).copy())
        self._eq_ids = np.array(
            [
                next(
                    j for j in range(m.neq) if m.eq_type[j] == 1 and m.body(m.eq_obj2id[j]).name.endswith("_" + f["id"])
                )
                for f in self.manifest["fruits"]
            ]
        )
        self._spring_ids = np.array([j for j, q, v, param in self._calibrated_joints])
        self._spring_qids = np.array([q + i for j, q, v, param in self._calibrated_joints for i in range(4)])
        plant_dofs = []
        for j in range(m.njnt):
            if "tomato_plant" in m.joint(j).name:
                v = m.jnt_dofadr[j]
                n = 6 if m.jnt_type[j] == 0 else 3 if m.jnt_type[j] == 1 else 1
                plant_dofs.extend(range(v, v + n))
        self._plant_dofs = np.array(plant_dofs)
        self._spring_k = m.jnt_stiffness[self._spring_ids].copy()
        self._spring_q = m.qpos_spring[self._spring_qids].copy()
        self._plant_damping = m.dof_damping[self._plant_dofs].copy()
        self._plant_armature = m.dof_armature[self._plant_dofs].copy()
        NewtonManager.register_state_force_callback(self._maintain_calibration)
        self._solver_ready = True

    def _maintain_calibration(self, state):
        # IsaacLab actuator property notifications reset ball spring references to
        # identity in this Newton revision. Preserve the authored intrinsic bend.
        solver = NewtonManager._solver
        m = solver.mj_model
        if getattr(self, "_reset_pending", False):
            solver._update_mjc_data(
                solver.mj_data if solver.use_mujoco_cpu else solver.mjw_data, NewtonManager.get_model(), state
            )
            self._reset_pending = False
        m.jnt_stiffness[self._spring_ids] = self._spring_k
        m.qpos_spring[self._spring_qids] = self._spring_q
        m.dof_damping[self._plant_dofs] = self._plant_damping
        m.dof_armature[self._plant_dofs] = self._plant_armature
        if not solver.use_mujoco_cpu:
            for field in ("jnt_stiffness", "qpos_spring", "dof_damping", "dof_armature"):
                arr = getattr(solver.mjw_model, field)
                arr.assign(np.broadcast_to(getattr(m, field), arr.numpy().shape).copy())

    def _check_separation(self):
        if not self._solver_ready:
            self._configure_solver()
        solver = NewtonManager._solver
        dt = self._step_dt
        self._elapsed += dt
        if solver.use_mujoco_cpu:
            d = solver.mj_data
            typ, ids, force = d.efc_type[: d.nefc], d.efc_id[: d.nefc], d.efc_force[: d.nefc]
            active = d.eq_active.copy()
        else:
            d = solver.mjw_data
            count = int(d.nefc.numpy()[0])
            typ, ids, force = (getattr(d.efc, f).numpy()[0, :count] for f in ("type", "id", "force"))
            active = d.eq_active.numpy()
        changed = False
        for f, eq in enumerate(self._eq_ids):
            if self.detached[f]:
                continue
            reaction = np.linalg.norm(force[(typ == 0) & (ids == eq)][:3])
            self.filtered_force[f] += (reaction - self.filtered_force[f]) * (
                1 - np.exp(-dt / self.cfg.force_filter_tau)
            )
            self._dwell[f] = self._dwell[f] + dt if self.filtered_force[f] > self.thresholds[f] else 0
            if self._dwell[f] >= self.cfg.separation_dwell:
                self.detached[f] = True
                active.reshape(-1)[eq] = False
                changed = True
                self.release_events.append(
                    {
                        "fruit": self.manifest["fruits"][f]["id"],
                        "time_s": self._elapsed,
                        "force_n": float(self.filtered_force[f]),
                    }
                )
        if changed:
            if solver.use_mujoco_cpu:
                d.eq_active[:] = active
            else:
                d.eq_active.assign(active)

    def _apply_wind(self, state):
        # CPU reference adapter; no aerodynamic forces are applied to released fruit.
        velocity = state.body_qd.numpy()[self._leaf_ids, :3]
        rel = np.asarray(self.cfg.wind_velocity) - velocity
        f = state.body_f.numpy()
        f[self._leaf_ids, :3] += (
            0.5 * 1.225 * 1.1 * self._leaf_areas[:, None] * np.linalg.norm(rel, axis=1)[:, None] * rel
        )
        state.body_f.assign(f)

    def _bind_stems(self, model):
        transforms = model.shape_transform.numpy()
        for shape, path in self._stem_shapes:
            mesh = model.shape_source[shape]
            xf = transforms[shape]
            rotation = Rotation.from_quat(xf[3:])
            world = rotation.apply(mesh.vertices) + xf[:3]
            bodies = self._path_bodies[path]
            distances = np.linalg.norm(world[:, None] - self._rest_q[bodies, :3], axis=-1)
            near = np.argsort(distances, axis=1)[:, :2]
            ids = bodies[near]
            weights = 1 / np.maximum(np.take_along_axis(distances, near, axis=1), 0.001) ** 4
            weights /= weights.sum(1, keepdims=True)
            rr = Rotation.from_quat(self._rest_q[ids, 3:].reshape(-1, 4))
            local = rr.inv().apply((world[:, None] - self._rest_q[ids, :3]).reshape(-1, 3)).reshape(ids.shape + (3,))
            self._bindings.append((shape, mesh, xf, ids, weights, local))

    def reset(self, env_ids=None, env_mask=None):
        if not self.is_initialized:
            return
        if env_mask is not None and not bool(torch.as_tensor(env_mask).any()):
            return
        if env_ids is not None and len(env_ids) == 0:
            return
        if not self._solver_ready:
            self._configure_solver()
        self.detached[:] = False
        self.filtered_force[:] = 0
        self._dwell[:] = 0
        self.release_events.clear()
        self._elapsed = 0
        self._reset_pending = True
        solver = NewtonManager._solver
        if solver.use_mujoco_cpu:
            solver.mj_data.eq_active[self._eq_ids] = True
        else:
            active = solver.mjw_data.eq_active.numpy()
            active[:, self._eq_ids] = True
            solver.mjw_data.eq_active.assign(active)
        # Scene resets reset the complete single world through NewtonManager. Restore
        # plant coordinates here too so a direct plant.reset() has the same semantics.
        model = NewtonManager.get_model()
        joint_bodies = model.joint_child.numpy()
        qstarts = model.joint_q_start.numpy()
        vstarts = model.joint_qd_start.numpy()
        state = NewtonManager._state_0
        q, v = state.joint_q.numpy(), state.joint_qd.numpy()
        for j, body in enumerate(joint_bodies):
            if body in self._plant_body_ids:
                q[qstarts[j] : qstarts[j + 1]] = self._rest_joint_q[qstarts[j] : qstarts[j + 1]]
                v[vstarts[j] : vstarts[j + 1]] = 0
        state.joint_q.assign(q)
        state.joint_qd.assign(v)
        solver._update_mjc_data(solver.mj_data if solver.use_mujoco_cpu else solver.mjw_data, model, state)

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        if not self.is_initialized or not self._bindings:
            return
        sim = SimulationContext.instance()
        viewers = [getattr(v, "_viewer", None) for v in sim._visualizers]
        viewers = [v for v in viewers if v is not None]
        if not viewers:
            return
        from .rendering import configure_greenhouse_lighting

        for viewer in viewers:
            configure_greenhouse_lighting(viewer)
        for viewer in viewers:
            if getattr(viewer, "_tomato_stem_streaming", False):
                continue
            original_log_state = viewer.log_state

            def log_state(state, original=original_log_state, target=viewer):
                original(state)
                # Mesh uploads must be between begin_frame and end_frame. RTX
                # clears queued edits at begin_frame, unlike rigid body poses.
                self._render_stems(target, state)

            viewer.log_state = log_state
            viewer._tomato_stem_streaming = True
            viewer.register_ui_callback(self._picking_ui, position="side")

    def _picking_ui(self, ui):
        ui.separator()
        ui.text("Tomato picking - red target")
        ui.text("DETACHED" if self.detached[0] else "Attached: close fingers, then pull away")
        ui.text(f"Load: {self.filtered_force[0]:.2f} / {self.thresholds[0]:.2f} N")
        ui.text(f"Time above threshold: {1000 * self._dwell[0]:.0f} / {1000 * self.cfg.separation_dwell:.0f} ms")
        ui.text(f"Fruit detached: {int(self.detached.sum())} / 16")

    def _render_stems(self, viewer, state):
        """Upload skinned stem vertices [m] inside the active viewer frame."""
        q = state.body_q.numpy()
        model = NewtonManager.get_model()
        # OVRTX runtime edits to a hidden USD prototype do not propagate to its
        # visible internal references. Stream those mesh instances directly.
        if hasattr(viewer, "stage") and not hasattr(viewer, "_tomato_stem_instances"):
            instances = {}
            for prim in viewer.stage.Traverse():
                references = prim.GetMetadata("references")
                if references:
                    for reference in references.GetAddedOrExplicitItems():
                        if not reference.assetPath:
                            instances.setdefault(str(reference.primPath), []).append(
                                str(prim.GetPath()).removeprefix("/root")
                            )
            viewer._tomato_stem_instances = instances
        for shape, mesh, xf, ids, weights, local in self._bindings:
            rot = Rotation.from_quat(q[ids, 3:].reshape(-1, 4))
            world = rot.apply(local.reshape(-1, 3)).reshape(local.shape) + q[ids, :3]
            world = (world * weights[..., None]).sum(axis=1)
            vertices = Rotation.from_quat(xf[3:]).inv().apply(world - xf[:3]).astype(np.float32)
            key = viewer._hash_geometry(
                int(model.shape_type.numpy()[shape]),
                tuple(model.shape_scale.numpy()[shape]),
                float(model.shape_margin.numpy()[shape]),
                bool(model.shape_is_solid.numpy()[shape]),
                mesh,
                False,
            )
            name = viewer._geometry_cache.get(key)
            if name:
                faces = np.asarray(mesh.indices).reshape(-1, 3)
                tri = vertices[faces]
                face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                normals = np.zeros_like(vertices)
                for corner in range(3):
                    np.add.at(normals, faces[:, corner], face_normals)
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
                targets = (
                    viewer._tomato_stem_instances.get(viewer._get_path(name), [])
                    if hasattr(viewer, "_tomato_stem_instances")
                    else [name]
                )
                for target in targets:
                    viewer.log_mesh(
                        target,
                        wp.array(vertices, dtype=wp.vec3, device=model.device),
                        wp.array(mesh.indices, dtype=wp.int32, device=model.device),
                        normals=wp.array(normals, dtype=wp.vec3, device=model.device),
                        hidden=target == name,
                        color=(0.12, 0.22, 0.035),
                        roughness=0.7,
                        dynamic=False,
                    )

    def observation(self):
        q = NewtonManager._state_0.body_q.numpy()[self.fruit_body_ids, :3]
        values = np.c_[q, self.filtered_force, self.detached.astype(np.float32)].reshape(1, -1)
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)
