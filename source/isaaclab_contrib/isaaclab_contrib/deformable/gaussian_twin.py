# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton deformables with an optional Gaussian-splat visual twin.

The Newton asset format stores a Gaussian field and a TetMesh in the same USD
package.  The ``newton:deformableSkin`` attributes on the Gaussian prim bind
each Gaussian center to TetMesh vertices.  This module keeps the physics mesh
in the regular Isaac Lab deformable-object path and updates the render-only
Gaussian transforms after every scene update.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp
from isaaclab_newton.physics import NewtonManager as SimulationManager

from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials import spawn_physics_material
from isaaclab.sim.utils import bind_physics_material, clone, get_current_stage, has_deformable_body_api
from isaaclab.utils.configclass import configclass

from .deformable_object import DeformableObject

if TYPE_CHECKING:
    from pxr import Usd

SKIN_NAMESPACE = "newton:deformableSkin"
_DEXSUITE_SIMULATION_MODES = {"stable-kinematic", "fast-kinematic"}


def _tetmesh_surface_triangles(tet_indices: np.ndarray) -> np.ndarray:
    """Return the boundary triangles of a tetrahedral mesh.

    The rigid benchmark uses this low-resolution boundary as a convex collision
    proxy. Interior tetrahedron faces must be removed; otherwise they make the
    convexifier needlessly expensive and can produce non-manifold input.
    """
    tetrahedra = np.asarray(tet_indices, dtype=np.int32).reshape(-1, 4)
    faces: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    for a, b, c, d in tetrahedra:
        for face in ((a, c, b), (a, b, d), (a, d, c), (b, c, d)):
            oriented = tuple(int(index) for index in face)
            key = tuple(sorted(oriented))
            faces[key] = None if key in faces else oriented
    surface = [face for face in faces.values() if face is not None]
    if not surface:
        raise ValueError("Gaussian twin TetMesh has no boundary triangles for rigid collision.")
    return np.asarray(surface, dtype=np.int32)


@clone
def spawn_gaussian_twin_rigid(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a Gaussian-twin package as one MJWarp rigid body.

    A bare ``UsdGeom.TetMesh`` is imported by Newton as a soft body even
    without deformable schemas. This spawner replaces it with a static copy of
    its boundary as a convex collision mesh, then attaches the package's
    Gaussian field to the enclosing rigid body. The Gaussian renderer thereby
    follows rigid-body motion without dynamic skinning updates.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    root_prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = get_current_stage()
    # The authored VisualMesh is useful to inspect package contents but would
    # otherwise occlude the Gaussian field in the rigid benchmark.
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdGeom.Imageable(prim).MakeInvisible()
    tet_prim = next((prim for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdGeom.TetMesh)), None)
    if tet_prim is None:
        raise ValueError(f"Gaussian twin asset at '{prim_path}' has no TetMesh prim for rigid collision.")

    tet_mesh = UsdGeom.TetMesh(tet_prim)
    points = np.asarray(tet_mesh.GetPointsAttr().Get(), dtype=np.float32)
    tet_indices = np.asarray(tet_mesh.GetTetVertexIndicesAttr().Get(), dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3 or tet_indices.size == 0:
        raise ValueError(f"Gaussian twin TetMesh '{tet_prim.GetPath()}' has invalid rigid-collision geometry.")
    surface = _tetmesh_surface_triangles(tet_indices)

    def _author_surface_mesh(path: str) -> UsdGeom.Mesh:
        """Author the extracted surface with a consistent triangle topology."""
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr().Set(points)
        mesh.CreateFaceVertexCountsAttr().Set([3] * len(surface))
        mesh.CreateFaceVertexIndicesAttr().Set(surface.reshape(-1))
        mesh.CreateSubdivisionSchemeAttr().Set("none")
        return mesh

    # Keep physics and diagnostics as two distinct shapes. A convex collision
    # shape is not a dependable visual representation: Newton may replace it
    # by generated hulls. The visual copy preserves the original TetMesh
    # boundary and is selected by the viewer TetMesh checkbox.
    collision_path = f"{prim_path}/RigidCollisionMesh"
    collision_mesh = _author_surface_mesh(collision_path)
    UsdGeom.Imageable(collision_mesh).MakeInvisible()
    collision_prim = collision_mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(collision_prim)
    UsdPhysics.MeshCollisionAPI.Apply(collision_prim).CreateApproximationAttr().Set("convexHull")

    visual_mesh = _author_surface_mesh(f"{prim_path}/RigidTetMeshVisual")
    visual_mesh.CreateDisplayColorAttr().Set([(0.25, 0.7, 1.0)])
    UsdGeom.Imageable(visual_mesh).MakeInvisible()

    mass = float(getattr(cfg, "rigid_mass", 0.05))
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"Gaussian twin rigid mass must be finite and positive, got {mass}.")
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr().Set(mass)

    # Do this after extracting its boundary so Newton cannot interpret the
    # original TetMesh as a legacy volume deformable.
    tet_prim.SetActive(False)
    return root_prim


@configclass
class GaussianTwinRigidUsdFileCfg(UsdFileCfg):
    """USD package configuration for the Gaussian-twin rigid benchmark."""

    func = "{DIR}.gaussian_twin:spawn_gaussian_twin_rigid"

    rigid_mass: float = 0.05
    """Rigid-body mass of one Gaussian-twin package [kg]."""


@clone
def spawn_gaussian_twin_deformable(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a Gaussian twin while excluding an auxiliary capture floor mesh."""
    from pxr import Usd, UsdGeom

    # Compose and apply the standard non-deformable overrides first.  Delaying
    # deformable schema application lets us remove the package's preview Plane
    # before Isaac Lab searches for the single visual deformable mesh.
    spawn_cfg = copy.deepcopy(cfg)
    deformable_props = spawn_cfg.deformable_props
    physics_material = spawn_cfg.physics_material
    spawn_cfg.deformable_props = None
    spawn_cfg.physics_material = None
    root_prim = spawn_from_usd(
        prim_path,
        spawn_cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == "Plane" and prim.IsA(UsdGeom.Mesh):
            prim.SetActive(False)

    if deformable_props is not None:
        stage = get_current_stage()
        if has_deformable_body_api(root_prim):
            schemas.modify_deformable_body_properties(prim_path, deformable_props, stage)
        else:
            schemas.define_deformable_body_properties(prim_path, deformable_props, stage, "volume")
        if spawn_cfg.mass_props is not None:
            raise ValueError(
                "MassPropertiesCfg are not supported for deformable bodies and should be set through "
                "deformable_props with mass=<value>."
            )

    # The standard USD spawner binds materials while the package root has no
    # deformable API yet.  Apply the material after the schema so Newton can
    # resolve it on the deformable body.
    if physics_material is not None:
        stage = get_current_stage()
        material_path = (
            spawn_cfg.physics_material_path
            if spawn_cfg.physics_material_path.startswith("/")
            else f"{prim_path}/{spawn_cfg.physics_material_path}"
        )
        spawn_physics_material(material_path, physics_material, stage=stage)
        bind_physics_material(prim_path, material_path, stage=stage)

    return root_prim


@configclass
class GaussianTwinDeformableUsdFileCfg(UsdFileCfg):
    """USD package configuration for a deformable Gaussian twin."""

    func = "{DIR}.gaussian_twin:spawn_gaussian_twin_deformable"


def _env_flag(name: str) -> bool:
    """Return whether an environment flag is enabled."""
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _show_tetmesh() -> bool:
    """Return whether the Gaussian-twin simulation surface should be rendered."""
    # Keep the original spelling as a compatibility alias for existing runs.
    return _env_flag("ISAACLAB_GAUSSIAN_TWIN_SHOW_TETMESH") or _env_flag("ISAACLAB_SHOW_GAUSSIAN_TWIN_TETMESH")


_gaussian_twin_updates_enabled = not _env_flag("ISAACLAB_GAUSSIAN_TWIN_DISABLE_GAUSSIAN_UPDATE")


def gaussian_twin_updates_enabled() -> bool:
    """Return the live Gaussian-skinning state selected in the RTX viewer."""
    return _gaussian_twin_updates_enabled


def set_gaussian_twin_updates_enabled(enabled: bool) -> None:
    """Enable or disable Gaussian skinning without changing deformable physics."""
    global _gaussian_twin_updates_enabled
    _gaussian_twin_updates_enabled = bool(enabled)


def _optional_positive_env(name: str, default: float) -> float:
    """Read a positive floating-point environment override with a clear error."""
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive floating-point value, got {raw_value!r}.") from error
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _dexsuite_simulation_mode() -> str | None:
    """Return the selected DexSuite-compatible material profile, if any."""
    mode = os.environ.get("ISAACLAB_GAUSSIAN_TWIN_DEXSUITE_SIMULATION", "").strip().lower()
    if not mode:
        return None
    if mode not in _DEXSUITE_SIMULATION_MODES:
        choices = ", ".join(sorted(_DEXSUITE_SIMULATION_MODES))
        raise ValueError(f"ISAACLAB_GAUSSIAN_TWIN_DEXSUITE_SIMULATION must be one of {choices}, got '{mode}'.")
    return mode


def _fit_asset_to_simulation_transform(
    asset_points: np.ndarray, simulation_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the authored TetMesh coordinates to the imported simulation mesh.

    The USD importer applies stage units, up-axis conversion, and the spawn
    transform to the TetMesh.  The custom Gaussian skinning attributes remain
    authored in the package's coordinate system.  Deriving the shared affine
    map from the ordered TetMesh vertices is therefore the only safe way to
    put the Gaussian bind pose in the physics/world coordinate system.
    """
    asset_points = np.asarray(asset_points, dtype=np.float64)
    simulation_points = np.asarray(simulation_points, dtype=np.float64)
    if asset_points.shape != simulation_points.shape or asset_points.ndim != 2 or asset_points.shape[1] != 3:
        raise ValueError(
            "The imported TetMesh does not have the same ordered vertices as the Gaussian skinning binding."
        )

    augmented = np.column_stack((asset_points, np.ones(len(asset_points))))
    affine, _, rank, _ = np.linalg.lstsq(augmented, simulation_points, rcond=None)
    if rank != 4:
        raise ValueError("The TetMesh vertices cannot determine an asset-to-simulation transform.")
    linear = affine[:3]
    translation = affine[3]
    reconstructed = asset_points @ linear + translation
    extent = max(float(np.ptp(simulation_points, axis=0).max()), 1.0e-6)
    max_error = float(np.max(np.abs(reconstructed - simulation_points)))
    if max_error > 1.0e-4 * extent:
        raise ValueError(
            "The USD importer changed TetMesh geometry rather than applying a single stage/placement transform; "
            "the authored Gaussian skinning binding cannot be used safely."
        )

    # ``linear`` maps row vectors. Warp quaternion matrices map column
    # vectors, hence the transpose before extracting the closest proper
    # rotation.
    u, _, vh = np.linalg.svd(linear.T)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return linear.astype(np.float32), translation.astype(np.float32), rotation.astype(np.float32)


def _quat_mul(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Hamilton product for xyzw quaternions, broadcasting leading axes."""
    fx, fy, fz, fw = (first[..., index] for index in range(4))
    sx, sy, sz, sw = (second[..., index] for index in range(4))
    return np.stack(
        (
            fw * sx + fx * sw + fy * sz - fz * sy,
            fw * sy - fx * sz + fy * sw + fz * sx,
            fw * sz + fx * sy - fy * sx + fz * sw,
            fw * sw - fx * sx - fy * sy - fz * sz,
        ),
        axis=-1,
    )


def _bake_aligned_background_gaussian_scale(model) -> None:
    """Bake the aligned-background USD scale into this task's static splats.

    Newton's generic viewers intentionally receive Gaussian centers directly,
    rather than through the mesh-instancing scale path.  The aligned capture is
    a task-owned static Gaussian field, so bake its inherited USD scale here
    instead of changing rendering behavior for every Newton application.
    """
    import newton

    labels = list(getattr(model, "shape_label", ()) or ())
    if not labels:
        return
    shape_scales = model.shape_scale.numpy()
    changed = False
    for shape_index, (label, source) in enumerate(zip(labels, model.shape_source, strict=True)):
        if not str(label).startswith("/World/GaussianBackground/") or not isinstance(source, newton.Gaussian):
            continue
        scale = np.asarray(shape_scales[shape_index], dtype=np.float32)
        if np.allclose(scale, 1.0):
            continue
        if not np.allclose(scale, scale[0]):
            raise ValueError(
                "The Gaussian-twin task currently supports only uniform scale on the aligned Gaussian background; "
                f"got {tuple(float(value) for value in scale)} for '{label}'."
            )
        source_data = source.warp_data
        if source_data is None:
            raise RuntimeError(f"Aligned Gaussian background '{label}' was not finalized by Newton.")
        baked = newton.Gaussian(
            positions=np.asarray(source.positions, dtype=np.float32) * scale,
            rotations=np.asarray(source.rotations, dtype=np.float32),
            scales=np.asarray(source.scales, dtype=np.float32) * abs(float(scale[0])),
            opacities=np.asarray(source.opacities, dtype=np.float32),
            sh_coeffs=np.asarray(source.sh_coeffs, dtype=np.float32),
            sh_degree=source.sh_degree,
            min_response=source.min_response,
            sorting_mode=source.sorting_mode,
            do_not_cast_shadows=source.do_not_cast_shadows,
        )
        baked.finalize(device=source_data.transforms.device)
        model.shape_source[shape_index] = baked
        shape_scales[shape_index] = (1.0, 1.0, 1.0)
        changed = True
    if changed:
        wp.copy(model.shape_scale, wp.array(shape_scales, dtype=wp.vec3f, device=model.shape_scale.device))


@dataclass
class _GaussianBinding:
    indices: wp.array
    weights: wp.array
    tet_indices: wp.array
    tet_rest_basis_inv: wp.array
    tet_volume: wp.array
    vertex_inv_volume: wp.array
    vertex_gradient: wp.array
    rest_transforms: wp.array
    rest_scales: wp.array
    rest_particle_offset: int
    particle_offset: int
    particle_count: int
    transforms: wp.array
    scales: wp.array
    opacities: wp.array
    rest_opacities: wp.array
    zero_opacities: wp.array


@wp.kernel
def _accumulate_vertex_gradients(
    particle_q: wp.array[wp.vec3],
    particle_offset: wp.int32,
    tet_indices: wp.array2d[wp.int32],
    rest_basis_inv: wp.array[wp.mat33],
    tet_volume: wp.array[wp.float32],
    vertex_gradient: wp.array[wp.mat33],
):
    """Accumulate volume-weighted deformation gradients on TetMesh vertices."""
    tet = wp.tid()
    i0 = particle_offset + tet_indices[tet, 0]
    x0 = particle_q[i0]
    gradient = (
        wp.matrix_from_cols(
            particle_q[particle_offset + tet_indices[tet, 1]] - x0,
            particle_q[particle_offset + tet_indices[tet, 2]] - x0,
            particle_q[particle_offset + tet_indices[tet, 3]] - x0,
        )
        * rest_basis_inv[tet]
    )
    contribution = gradient * tet_volume[tet]
    for corner in range(4):
        wp.atomic_add(vertex_gradient, tet_indices[tet, corner], contribution)


@wp.kernel
def _skin_gaussians(
    particle_q: wp.array[wp.vec3],
    rest_particle_q: wp.array[wp.vec3],
    influence_indices: wp.array2d[wp.int32],
    influence_weights: wp.array2d[wp.float32],
    vertex_gradient: wp.array[wp.mat33],
    vertex_inv_volume: wp.array[wp.float32],
    rest_transforms: wp.array[wp.transformf],
    rest_scales: wp.array[wp.vec3f],
    rest_particle_offset: wp.int32,
    particle_offset: wp.int32,
    deform_scales: wp.int32,
    transforms: wp.array[wp.transformf],
    scales: wp.array[wp.vec3f],
):
    """Skin Gaussian centers, orientations, and optionally scales to a TetMesh."""
    gaussian = wp.tid()
    center = wp.transform_get_translation(rest_transforms[gaussian])
    displacement = wp.vec3(0.0, 0.0, 0.0)
    gradient = wp.mat33(0.0)
    for influence in range(influence_indices.shape[1]):
        local_vertex = influence_indices[gaussian, influence]
        particle_vertex = particle_offset + local_vertex
        rest_vertex = rest_particle_offset + local_vertex
        weight = influence_weights[gaussian, influence]
        displacement += weight * (particle_q[particle_vertex] - rest_particle_q[rest_vertex])
        gradient += weight * vertex_gradient[local_vertex] * vertex_inv_volume[local_vertex]

    rest = rest_transforms[gaussian]
    c0 = wp.vec3(gradient[0, 0], gradient[1, 0], gradient[2, 0])
    c1 = wp.vec3(gradient[0, 1], gradient[1, 1], gradient[2, 1])
    if wp.length(c0) < 1.0e-9 or wp.length(wp.cross(c0, c1)) < 1.0e-12:
        transforms[gaussian] = wp.transform(center + displacement, wp.transform_get_rotation(rest))
        if deform_scales != 0:
            scales[gaussian] = rest_scales[gaussian]
        return

    c0 = wp.normalize(c0)
    c1 = wp.normalize(c1 - c0 * wp.dot(c0, c1))
    rotation = wp.quat_from_matrix(wp.matrix_from_cols(c0, c1, wp.cross(c0, c1)))
    transforms[gaussian] = wp.transform(center + displacement, wp.normalize(rotation * wp.transform_get_rotation(rest)))
    if deform_scales != 0:
        # A Gaussian covariance represents principal radii but not shear. Keep
        # the local axial stretches from the blended deformation gradient; the
        # rotation above carries the orthonormal part.
        stretch = wp.vec3(
            wp.length(wp.vec3(gradient[0, 0], gradient[1, 0], gradient[2, 0])),
            wp.length(wp.vec3(gradient[0, 1], gradient[1, 1], gradient[2, 1])),
            wp.length(wp.vec3(gradient[0, 2], gradient[1, 2], gradient[2, 2])),
        )
        rest_scale = rest_scales[gaussian]
        scales[gaussian] = wp.vec3(rest_scale[0] * stretch[0], rest_scale[1] * stretch[1], rest_scale[2] * stretch[2])


@wp.kernel
def _skin_gaussian_positions(
    particle_q: wp.array[wp.vec3],
    rest_particle_q: wp.array[wp.vec3],
    influence_indices: wp.array2d[wp.int32],
    influence_weights: wp.array2d[wp.float32],
    rest_transforms: wp.array[wp.transformf],
    rest_particle_offset: wp.int32,
    particle_offset: wp.int32,
    transforms: wp.array[wp.transformf],
):
    """Skin only Gaussian centers, avoiding the deformation-gradient pass."""
    gaussian = wp.tid()
    displacement = wp.vec3(0.0, 0.0, 0.0)
    for influence in range(influence_indices.shape[1]):
        local_vertex = influence_indices[gaussian, influence]
        particle_vertex = particle_offset + local_vertex
        rest_vertex = rest_particle_offset + local_vertex
        displacement += influence_weights[gaussian, influence] * (
            particle_q[particle_vertex] - rest_particle_q[rest_vertex]
        )
    rest = rest_transforms[gaussian]
    transforms[gaussian] = wp.transform(
        wp.transform_get_translation(rest) + displacement,
        wp.transform_get_rotation(rest),
    )


@wp.kernel
def _reset_particles(
    rest_particle_q: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    rest_offset: wp.int32,
    particle_offset: wp.int32,
    particle_count: wp.int32,
    parked: wp.int32,
):
    index = wp.tid()
    if index >= particle_count:
        return
    position = rest_particle_q[rest_offset + index]
    if parked != 0:
        position += wp.vec3(0.0, 0.0, -100.0)
    particle_q[particle_offset + index] = position
    particle_qd[particle_offset + index] = wp.vec3(0.0, 0.0, 0.0)


class GaussianTwinDeformableObject(DeformableObject):
    """Newton deformable object that skins an authored Gaussian field to its TetMesh."""

    # Shared by the slots in one Newton model so all slot objects agree on
    # which toys are active during the same reset. The model identity keeps
    # separate environments in one Python process isolated.
    _selection_by_model_env: dict[tuple[int, int, int, int], np.ndarray] = {}
    _spawn_offsets_by_model_env: dict[tuple[int, int, int, int], np.ndarray] = {}
    _MODE_ATTRIBUTES: ClassVar[dict[str, tuple[str, ...]]] = {
        "position": ("positions",),
        "position-rotation": ("positions", "orientations"),
        "position-rotation-scale": ("positions", "orientations", "scales"),
    }

    def __init__(self, cfg):
        # The base deformable class establishes the actual instance count
        # during initialization, rather than in its constructor.
        self._num_instances = 0
        super().__init__(cfg)
        self._gaussian_bindings: list[_GaussianBinding] = []
        self._gaussian_initialized = False
        self._rest_particles: wp.array | None = None
        self._base_rest_particles: np.ndarray | None = None
        self._spawn_offsets: np.ndarray | None = None
        self._rest_particle_offsets: list[int] = []
        self._slot_index = int(getattr(cfg, "slot_index", 0))
        self._num_objects = int(getattr(cfg, "num_objects", 1))
        self._num_slots = int(getattr(cfg, "num_slots", 1))
        self._splat_deformation = str(getattr(cfg, "splat_deformation", "position-rotation-scale"))
        if self._splat_deformation not in self._MODE_ATTRIBUTES:
            choices = ", ".join(self._MODE_ATTRIBUTES)
            raise ValueError(
                f"Unknown Gaussian splat deformation '{self._splat_deformation}'; expected one of: {choices}."
            )

    def _initialize_impl(self):
        super()._initialize_impl()
        self._set_visual_mesh_visibility()
        self._initialize_gaussian_bindings()

    def _set_visual_mesh_visibility(self) -> None:
        """Hide the simulation surface so the Gaussian twin is the displayed object.

        The Newton reference example renders the TetMesh only as an optional debug view. The
        imported package also contains a regular ``VisualMesh``; leaving it visible makes the
        moving gray triangulated surface obscure the Gaussian field after the first simulation
        update. Set ``ISAACLAB_GAUSSIAN_TWIN_SHOW_TETMESH=1`` to inspect that
        surface. The original ``ISAACLAB_SHOW_GAUSSIAN_TWIN_TETMESH`` spelling
        remains supported.
        """
        if _show_tetmesh() or not gaussian_twin_updates_enabled():
            return

        import re

        from pxr import UsdGeom

        from isaaclab.sim import get_current_stage

        stage = get_current_stage()
        if stage is None:
            return
        for env_idx in range(self._num_instances):
            path = re.sub(r"(?<=[Ee]nv_)(?:\[\^/\][*+]|\.\*)", str(env_idx), self._registry_entry.vis_mesh_prim_path)
            path = re.sub(r"\[\^/\][*+]|\.\*", str(env_idx), path)
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()

    def _register_deformable(self):
        """Register a packaged TetMesh with scale-aware VBD material parameters."""
        entry = super()._register_deformable()
        # The DexSuite training presets use fixed, deliberately soft material
        # and a 4 mm collision shell.  This is a complete physics profile,
        # not merely a cheaper solver budget, so it must override the package
        # scale-aware material derived for the Gaussian-twin demonstration.
        if _dexsuite_simulation_mode() is not None:
            entry.density = 300.0
            entry.k_mu = 1.0e5
            entry.k_lambda = 1.0e5
            entry.k_damp = 1.0e-5
            entry.particle_radius = 0.004
            return entry
        from pxr import Usd

        stage = Usd.Stage.Open(self.cfg.spawn.usd_path)
        if stage is None:
            return entry
        for prim in stage.Traverse():
            if prim.GetTypeName() != "TetMesh":
                continue
            points = np.asarray(entry.vertices, dtype=np.float64)
            tet_indices = np.asarray(entry.indices, dtype=np.int32).reshape(-1, 4)
            corners = points[tet_indices]
            edge_length = float(
                np.mean(
                    [
                        np.linalg.norm(corners[:, first] - corners[:, second], axis=1)
                        for first, second in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
                    ]
                )
            )
            toy_length = float(np.ptp(points, axis=0).max())
            if edge_length <= 0.0 or toy_length <= 0.0:
                raise ValueError(f"Gaussian twin TetMesh '{prim.GetPath()}' has invalid physical dimensions.")

            # Match the Newton reference example: expressing Young's modulus as
            # E/(rho*g*L), and damping as an element damping ratio, makes a
            # 10 cm and a 30 cm packaged toy sag and settle comparably. Fixed
            # Pascal and Pa*s values were the source of the unstable fine-mesh
            # assets in this task.
            density = _optional_positive_env("ISAACLAB_GAUSSIAN_TWIN_DENSITY", float(entry.density))
            gravity_stiffness = _optional_positive_env("ISAACLAB_GAUSSIAN_TWIN_GRAVITY_STIFFNESS", 125.0)
            damping_ratio = _optional_positive_env("ISAACLAB_GAUSSIAN_TWIN_DAMPING_RATIO", 0.1)
            poisson = float(os.environ.get("ISAACLAB_GAUSSIAN_TWIN_POISSONS_RATIO", "0.35"))
            if not -1.0 < poisson < 0.5:
                raise ValueError(f"ISAACLAB_GAUSSIAN_TWIN_POISSONS_RATIO must be in (-1, 0.5), got {poisson}.")
            youngs_modulus = _optional_positive_env(
                "ISAACLAB_GAUSSIAN_TWIN_YOUNGS_MODULUS", gravity_stiffness * density * 9.81 * toy_length
            )
            entry.density = density
            entry.k_mu = youngs_modulus / (2.0 * (1.0 + poisson))
            entry.k_lambda = youngs_modulus * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
            entry.k_damp = 2.0 * damping_ratio * edge_length * np.sqrt(youngs_modulus * density)

            radius_attr = prim.GetAttribute("newton:simulationMesh:particleRadius")
            radius = radius_attr.Get() if radius_attr.IsValid() else None
            if radius is not None and float(radius) > 0.0:
                entry.particle_radius = float(radius)
            else:
                entry.particle_radius = 0.4 * edge_length
            break
        return entry

    def _initialize_gaussian_bindings(self) -> None:
        """Find imported Gaussian shapes and prepare their skinning buffers."""
        import newton

        from pxr import Usd

        stage = Usd.Stage.Open(self.cfg.spawn.usd_path)
        if stage is None:
            raise RuntimeError(f"Unable to open Gaussian twin asset '{self.cfg.spawn.usd_path}'.")
        gaussian_prims = [p for p in stage.Traverse() if p.GetTypeName() == "ParticleField3DGaussianSplat"]
        if not gaussian_prims:
            raise ValueError(
                f"Gaussian twin asset '{self.cfg.spawn.usd_path}' has no ParticleField3DGaussianSplat prim."
            )
        gaussian_prim = gaussian_prims[0]
        tet_prim = next((prim for prim in stage.Traverse() if prim.GetTypeName() == "TetMesh"), None)
        if tet_prim is None:
            raise ValueError(f"Gaussian twin asset '{self.cfg.spawn.usd_path}' has no TetMesh prim.")

        def required(name: str):
            attr = gaussian_prim.GetAttribute(f"{SKIN_NAMESPACE}:{name}")
            value = attr.Get() if attr.IsValid() else None
            if value is None:
                raise ValueError(f"{gaussian_prim.GetPath()} is missing '{SKIN_NAMESPACE}:{name}'.")
            return value

        influence_size = int(required("influenceSize"))
        if influence_size <= 0:
            raise ValueError(f"{gaussian_prim.GetPath()} has an invalid influenceSize of {influence_size}.")
        indices = np.asarray(required("influenceIndices"), dtype=np.int32).reshape(-1, influence_size)
        weights = np.asarray(required("influenceWeights"), dtype=np.float32).reshape(-1, influence_size)
        point_count = int(required("pointCount"))
        if point_count <= 0 or len(indices) != point_count or len(weights) != point_count:
            raise ValueError("Gaussian skinning arrays do not match the authored pointCount.")
        if not np.isfinite(weights).all():
            raise ValueError("Gaussian skinning weights must be finite.")
        tet_indices = np.asarray(tet_prim.GetAttribute("tetVertexIndices").Get(), dtype=np.int32).reshape(-1, 4)
        if len(tet_indices) == 0:
            raise ValueError(f"{tet_prim.GetPath()} has no tetrahedra to drive the Gaussian field.")
        asset_points = np.asarray(tet_prim.GetAttribute("points").Get(), dtype=np.float32)
        if len(asset_points) != self._particles_per_body:
            raise ValueError(
                f"{tet_prim.GetPath()} has {len(asset_points)} vertices, but Newton imported "
                f"{self._particles_per_body} deformable particles."
            )

        model = SimulationManager.get_model()
        state = SimulationManager.get_state_0()
        if model is None or state is None or state.particle_q is None:
            raise RuntimeError("Newton state is unavailable while initializing the Gaussian twin.")
        _bake_aligned_background_gaussian_scale(model)
        if indices.min() < 0 or indices.max() >= self._particles_per_body:
            raise ValueError("Gaussian skinning influences must index vertices of this asset's simulation TetMesh.")
        if tet_indices.min() < 0 or tet_indices.max() >= self._particles_per_body:
            raise ValueError("TetMesh connectivity must index vertices of this deformable object.")
        labels = getattr(model, "shape_label", [])
        sources = [
            (index, source)
            for index, source in enumerate(model.shape_source)
            if isinstance(source, newton.Gaussian)
            and (not labels or f"GaussianTwin_{self._slot_index}" in str(labels[index]))
        ]
        if not sources:
            raise RuntimeError("The Newton model contains no imported Gaussian shape.")
        if len(sources) != self._num_instances:
            raise RuntimeError(
                f"Expected one Gaussian shape per environment ({self._num_instances}), found {len(sources)}. "
                "Use num_envs=1 or enable Newton Gaussian replication for this asset."
            )

        # The registry records the flat particle range for each environment.
        self._rest_particles = wp.array(
            self._default_nodal_pos_w.numpy().reshape(-1, 3), dtype=wp.vec3, device=self.device
        )
        self._base_rest_particles = self._default_nodal_pos_w.numpy().reshape(-1, 3).copy()
        self._spawn_offsets = np.zeros((self._num_instances, 3), dtype=np.float32)
        shape_transforms = model.shape_transform.numpy()
        shape_scales = model.shape_scale.numpy()
        shape_bodies = model.shape_body.numpy()
        for env_idx, (shape_index, source) in enumerate(sources):
            if int(shape_bodies[shape_index]) >= 0:
                raise RuntimeError("Gaussian twins attached to dynamic rigid bodies are not supported.")
            source._newton_dynamic_attributes = self._MODE_ATTRIBUTES[self._splat_deformation]
            offset = int(self._particle_offsets.numpy()[env_idx])
            rest_points = self._default_nodal_pos_w.numpy()[env_idx]
            asset_linear, asset_translation, asset_rotation = _fit_asset_to_simulation_transform(
                asset_points, rest_points
            )
            source_data = source.warp_data
            rest = np.asarray(source_data.transforms.numpy(), dtype=np.float32).copy()
            singular_values = np.linalg.svd(asset_linear, compute_uv=False)
            uniform_scale = float(np.cbrt(abs(np.linalg.det(asset_linear))))
            if not np.allclose(singular_values, uniform_scale, rtol=1.0e-4, atol=1.0e-6):
                raise ValueError(
                    "Gaussian-twin assets currently require a uniform asset-to-simulation scale; "
                    f"got singular values {tuple(float(value) for value in singular_values)}."
                )
            rest_scales = np.asarray(source_data.scales.numpy(), dtype=np.float32).copy() * uniform_scale
            rotation_quat = np.asarray(
                wp.quat_from_matrix(wp.mat33(*asset_rotation.reshape(-1).tolist())), dtype=np.float32
            )
            rest[:, :3] = rest[:, :3] @ asset_linear + asset_translation
            rest[:, 3:7] = _quat_mul(rotation_quat, rest[:, 3:7])

            # The Gaussian transforms now contain the complete stage conversion
            # and placement.  Resetting the shape transform removes the second
            # application of that transform in the renderer, and also means each
            # preloaded slot has an independent world-space bind pose.
            wp.copy(source_data.transforms, wp.array(rest, dtype=wp.transformf, device=self.device))
            wp.copy(source_data.scales, wp.array(rest_scales, dtype=wp.vec3f, device=self.device))
            shape_transforms[shape_index] = np.asarray(wp.transform_identity(), dtype=np.float32)
            shape_scales[shape_index] = (1.0, 1.0, 1.0)
            basis_inv, tet_volume, vertex_inv_volume = self._tet_rest_data(rest_points, tet_indices)
            self._rest_particle_offsets.append(env_idx * self._particles_per_body)
            self._gaussian_bindings.append(
                _GaussianBinding(
                    indices=wp.array(indices, dtype=wp.int32, device=self.device),
                    weights=wp.array(weights, dtype=wp.float32, device=self.device),
                    tet_indices=wp.array(tet_indices, dtype=wp.int32, device=self.device),
                    tet_rest_basis_inv=wp.array(basis_inv, dtype=wp.mat33, device=self.device),
                    tet_volume=wp.array(tet_volume, dtype=wp.float32, device=self.device),
                    vertex_inv_volume=wp.array(vertex_inv_volume, dtype=wp.float32, device=self.device),
                    vertex_gradient=wp.zeros(self._particles_per_body, dtype=wp.mat33, device=self.device),
                    rest_transforms=wp.array(rest, dtype=wp.transformf, device=self.device),
                    rest_scales=wp.array(rest_scales, dtype=wp.vec3f, device=self.device),
                    rest_particle_offset=env_idx * self._particles_per_body,
                    particle_offset=offset,
                    particle_count=len(rest),
                    transforms=source_data.transforms,
                    scales=source_data.scales,
                    opacities=source_data.opacities,
                    rest_opacities=wp.array(source_data.opacities.numpy(), dtype=wp.float32, device=self.device),
                    zero_opacities=wp.zeros(len(source_data.opacities), dtype=wp.float32, device=self.device),
                )
            )
        wp.copy(model.shape_transform, wp.array(shape_transforms, dtype=wp.transform, device=self.device))
        wp.copy(model.shape_scale, wp.array(shape_scales, dtype=wp.vec3f, device=self.device))
        self._gaussian_initialized = True

    @staticmethod
    def _tet_rest_data(points: np.ndarray, tet_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build volume-weighted rest deformation-gradient data for a TetMesh."""
        corners = points[tet_indices]
        edges = np.stack([corners[:, corner] - corners[:, 0] for corner in (1, 2, 3)], axis=-1)
        determinant = np.linalg.det(edges)
        usable = np.abs(determinant) > 1.0e-16
        basis_inv = np.zeros_like(edges)
        basis_inv[usable] = np.linalg.inv(edges[usable])
        volume = np.where(usable, np.abs(determinant) / 6.0, 0.0)
        accumulated = np.zeros(len(points), dtype=np.float64)
        np.add.at(accumulated, tet_indices.ravel(), np.repeat(volume, 4))
        inv_volume = np.where(accumulated > 0.0, 1.0 / accumulated, 0.0)
        return basis_inv.astype(np.float32), volume.astype(np.float32), inv_volume.astype(np.float32)

    def _selection(self, env_id: int, *, refresh: bool) -> np.ndarray:
        """Return this reset's shared slot selection for one environment."""
        model = SimulationManager.get_model()
        key = (id(model), self._num_slots, self._num_objects, env_id)
        selection = self._selection_by_model_env.get(key)
        # Slot zero is reset first by the scene in insertion order. Refreshing
        # there gives every slot a new toy selection every episode, while the
        # other slots read the same draw. ``np.random`` follows Isaac Lab's
        # process seed; ``default_rng()`` would seed from OS entropy instead.
        if refresh or selection is None:
            selection = np.random.choice(self._num_slots, size=self._num_objects, replace=False)
            self._selection_by_model_env[key] = selection
        return selection

    def _spawn_offset(self, env_id: int, *, refresh: bool) -> np.ndarray:
        """Return a shared per-slot XY reset jitter for one environment."""
        model = SimulationManager.get_model()
        key = (id(model), self._num_slots, self._num_objects, env_id)
        offsets = self._spawn_offsets_by_model_env.get(key)
        if refresh or offsets is None:
            # Keep the movement inside the reachable tabletop workspace. The
            # slot layout is already separated; this adds episode variation
            # without requiring the expensive global inter-object contact path.
            offsets = np.zeros((self._num_slots, 3), dtype=np.float32)
            offsets[:, 0] = np.random.uniform(-0.06, 0.06, size=self._num_slots)
            offsets[:, 1] = np.random.uniform(-0.08, 0.08, size=self._num_slots)
            self._spawn_offsets_by_model_env[key] = offsets
        return offsets[self._slot_index]

    def _apply_spawn_offset(self, env_id: int, offset: np.ndarray, binding: _GaussianBinding) -> None:
        """Move the TetMesh rest pose and its Gaussian bind pose together."""
        if self._base_rest_particles is None or self._spawn_offsets is None:
            return
        delta = offset - self._spawn_offsets[env_id]
        self._spawn_offsets[env_id] = offset
        rest_particles = self._base_rest_particles.copy()
        for index, current in enumerate(self._spawn_offsets):
            start = index * self._particles_per_body
            rest_particles[start : start + self._particles_per_body] += current
        wp.copy(self._rest_particles, wp.array(rest_particles, dtype=wp.vec3, device=self.device))
        rest = np.asarray(binding.rest_transforms.numpy(), dtype=np.float32).copy()
        rest[:, :3] += delta
        wp.copy(binding.rest_transforms, wp.array(rest, dtype=wp.transformf, device=self.device))
        wp.copy(binding.transforms, wp.array(rest, dtype=wp.transformf, device=self.device))

    def reset(self, env_ids=None, env_mask=None) -> None:
        super().reset(env_ids, env_mask)
        if not self._gaussian_initialized or self._rest_particles is None:
            return
        if env_ids is None:
            if env_mask is None:
                env_ids = range(self._num_instances)
            else:
                mask = np.asarray(env_mask.numpy() if hasattr(env_mask, "numpy") else env_mask, dtype=bool).reshape(-1)
                if len(mask) != self._num_instances:
                    raise ValueError(f"env_mask must have {self._num_instances} entries, got {len(mask)}.")
                env_ids = np.flatnonzero(mask)
        elif env_mask is not None:
            mask = np.asarray(env_mask.numpy() if hasattr(env_mask, "numpy") else env_mask, dtype=bool).reshape(-1)
            env_ids = [env_id for env_id in env_ids if mask[int(env_id)]]
        state_0 = SimulationManager.get_state_0()
        state_1 = SimulationManager.get_state_1()
        for env_id in (int(value) for value in env_ids):
            active = self._slot_index in self._selection(env_id, refresh=self._slot_index == 0)
            binding = self._gaussian_bindings[env_id]
            self._apply_spawn_offset(
                env_id,
                self._spawn_offset(env_id, refresh=self._slot_index == 0),
                binding,
            )
            for state in (state_0, state_1):
                if state is None or state.particle_q is None or state.particle_qd is None:
                    continue
                wp.launch(
                    _reset_particles,
                    dim=self._particles_per_body,
                    inputs=[
                        self._rest_particles,
                        state.particle_q,
                        state.particle_qd,
                        self._rest_particle_offsets[env_id],
                        binding.particle_offset,
                        self._particles_per_body,
                        0 if active else 1,
                    ],
                    device=self.device,
                )
            if active:
                wp.copy(binding.opacities, binding.rest_opacities)
            else:
                wp.copy(binding.opacities, binding.zero_opacities)

    def update(self, dt: float):
        super().update(dt)
        # The RTX viewer can toggle this live, immediately stopping skinning
        # and GPU attribute streaming while preserving deformable physics.
        if not self._gaussian_initialized or not gaussian_twin_updates_enabled():
            return
        state = SimulationManager.get_state_0()
        if state is None or state.particle_q is None:
            return
        rest_particles = self._rest_particles
        for binding in self._gaussian_bindings:
            # Bind poses have been converted to world space above, then the
            # dynamic source arrays are replaced in place for Newton RTX.
            if self._splat_deformation == "position":
                wp.launch(
                    _skin_gaussian_positions,
                    dim=binding.particle_count,
                    inputs=[
                        state.particle_q,
                        rest_particles,
                        binding.indices,
                        binding.weights,
                        binding.rest_transforms,
                        binding.rest_particle_offset,
                        binding.particle_offset,
                    ],
                    outputs=[binding.transforms],
                    device=self.device,
                )
                continue

            binding.vertex_gradient.zero_()
            wp.launch(
                _accumulate_vertex_gradients,
                dim=len(binding.tet_volume),
                inputs=[
                    state.particle_q,
                    binding.particle_offset,
                    binding.tet_indices,
                    binding.tet_rest_basis_inv,
                    binding.tet_volume,
                ],
                outputs=[binding.vertex_gradient],
                device=self.device,
            )
            wp.launch(
                _skin_gaussians,
                dim=binding.particle_count,
                inputs=[
                    state.particle_q,
                    rest_particles,
                    binding.indices,
                    binding.weights,
                    binding.vertex_gradient,
                    binding.vertex_inv_volume,
                    binding.rest_transforms,
                    binding.rest_scales,
                    binding.rest_particle_offset,
                    binding.particle_offset,
                    int(self._splat_deformation == "position-rotation-scale"),
                ],
                outputs=[binding.transforms, binding.scales],
                device=self.device,
            )
