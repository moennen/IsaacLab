# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless Newton deformable spawners for the experimental Dexsuite task."""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np

from isaaclab.sim.spawners.materials.physics_materials_cfg import DeformableBodyMaterialBaseCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import VisualMaterialCfg
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.version import has_kit

if TYPE_CHECKING:
    from pxr import Usd


def _clone(func: Callable) -> Callable:
    """Defer IsaacLab USD helper imports until spawning time."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from isaaclab.sim.utils import clone

        return clone(func)(*args, **kwargs)

    return wrapper


@configclass
class NewtonTetCuboidCfg(DeformableObjectSpawnerCfg):
    """Pre-tetrahedralized cuboid spawner for Newton deformable bodies.

    The generic ``MeshCuboidCfg(deformable_props=...)`` path tetrahedralizes
    through ``omni.physx``.  This spawner authors the small ``UsdGeom.TetMesh``
    directly, which keeps the task runnable from a kitless Newton install.
    """

    func: Callable | str = "{DIR}.spawners:spawn_newton_tet_cuboid"

    size: tuple[float, float, float] = MISSING
    """Cuboid side lengths [m]."""

    resolution: tuple[int, int, int] = (3, 3, 2)
    """Number of grid cells along x/y/z before tetrahedral splitting."""

    visual_material_path: str = "visual_material"
    """Relative path for the visual material."""

    visual_material: VisualMaterialCfg | None = None
    """Optional visual material."""

    physics_material_path: str = "physics_material"
    """Relative path for the Newton deformable material."""

    physics_material: DeformableBodyMaterialBaseCfg = MISSING
    """Newton deformable material."""


@configclass
class NewtonVbdTetAssetCfg(DeformableObjectSpawnerCfg):
    """Spawner for legacy VBD tetrahedral assets authored with custom USD attributes."""

    func: Callable | str = "{DIR}.spawners:spawn_newton_vbd_tet_asset"

    usd_path: str = MISSING
    """Path to the USD file containing ``vbd:vertices`` and ``vbd:tet_indices`` attributes."""

    source_prim_path: str = "/TetMesh"
    """Prim path in :attr:`usd_path` containing the VBD custom attributes."""

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Per-axis scale applied after source-axis conversion."""

    rotate_y_up_to_z_up: bool = True
    """Rotate source Y-up coordinates into Isaac Lab's Z-up world frame."""

    center_to_origin: bool = True
    """Recentre vertices around their mean so ``init_state.pos`` is the deformable COM."""

    visual_material_path: str = "visual_material"
    """Relative path for the visual material."""

    visual_material: VisualMaterialCfg | None = None
    """Optional visual material."""

    physics_material_path: str = "physics_material"
    """Relative path for the Newton deformable material."""

    physics_material: DeformableBodyMaterialBaseCfg = MISSING
    """Newton deformable material."""


def _surface_faces_from_tets(tets: np.ndarray) -> np.ndarray:
    """Return triangle faces that belong to only one tet."""
    face_counts: Counter[tuple[int, int, int]] = Counter()
    oriented_faces: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for tet in tets:
        tet_faces = (
            (int(tet[0]), int(tet[2]), int(tet[1])),
            (int(tet[0]), int(tet[1]), int(tet[3])),
            (int(tet[1]), int(tet[2]), int(tet[3])),
            (int(tet[2]), int(tet[0]), int(tet[3])),
        )
        for face in tet_faces:
            key = tuple(sorted(face))
            face_counts[key] += 1
            oriented_faces[key] = face

    return np.asarray([oriented_faces[face] for face, count in face_counts.items() if count == 1], dtype=np.int32)


def _ensure_positive_tet_winding(vertices: np.ndarray, tets: np.ndarray) -> None:
    """Flip tetrahedra in-place so all signed volumes are positive."""
    points = vertices[tets]
    signed_volume = np.einsum(
        "ij,ij->i",
        points[:, 1] - points[:, 0],
        np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]),
    )
    degenerate = np.isclose(signed_volume, 0.0)
    if np.any(degenerate):
        raise ValueError(f"VBD tet asset contains {int(np.count_nonzero(degenerate))} degenerate tetrahedra.")

    negative = signed_volume < 0.0
    if np.any(negative):
        flipped = tets[negative, 2].copy()
        tets[negative, 2] = tets[negative, 3]
        tets[negative, 3] = flipped


def _vbd_tet_asset_geometry(
    usd_path: str,
    source_prim_path: str = "/TetMesh",
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotate_y_up_to_z_up: bool = True,
    center_to_origin: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a legacy VBD tet asset and return vertices, tetrahedra, and surface triangles."""
    from pxr import Usd

    source_stage = Usd.Stage.Open(str(usd_path))
    if source_stage is None:
        raise FileNotFoundError(f"Failed to open VBD tet asset: '{usd_path}'.")

    source_prim = source_stage.GetPrimAtPath(source_prim_path)
    if not source_prim.IsValid():
        raise ValueError(f"Could not find prim '{source_prim_path}' in VBD tet asset '{usd_path}'.")

    vertices_attr = source_prim.GetAttribute("vbd:vertices")
    tet_indices_attr = source_prim.GetAttribute("vbd:tet_indices")
    if not vertices_attr.IsValid() or not tet_indices_attr.IsValid():
        raise ValueError(
            f"Prim '{source_prim_path}' in '{usd_path}' must define vbd:vertices and vbd:tet_indices attributes."
        )

    vertices = np.asarray(vertices_attr.Get(), dtype=np.float32)
    raw_tets = np.asarray(tet_indices_attr.Get(), dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vbd:vertices to have shape (N, 3), got {vertices.shape}.")
    if raw_tets.ndim != 1 or raw_tets.size % 4 != 0:
        raise ValueError(f"Expected flat vbd:tet_indices with a multiple of four entries, got {raw_tets.shape}.")

    tets = raw_tets.reshape(-1, 4).copy()
    if np.any(tets < 0) or np.any(tets >= vertices.shape[0]):
        raise ValueError("VBD tet asset contains tetrahedron indices outside the vertex range.")

    if rotate_y_up_to_z_up:
        vertices = np.stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]), axis=1)

    vertices = vertices * np.asarray(scale, dtype=np.float32).reshape(1, 3)
    if center_to_origin:
        vertices = vertices - vertices.mean(axis=0, keepdims=True)

    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    _ensure_positive_tet_winding(vertices, tets)
    return vertices, tets, _surface_faces_from_tets(tets)


def _cuboid_tet_grid(
    size: tuple[float, float, float], resolution: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = (int(v) for v in resolution)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Cuboid tet resolution must be positive, got {resolution}.")

    sx, sy, sz = (0.5 * float(v) for v in size)
    xs = np.linspace(-sx, sx, nx + 1, dtype=np.float32)
    ys = np.linspace(-sy, sy, ny + 1, dtype=np.float32)
    zs = np.linspace(-sz, sz, nz + 1, dtype=np.float32)

    vertices = np.asarray([[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float32)

    def vid(i: int, j: int, k: int) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    tets = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v000 = vid(i, j, k)
                v100 = vid(i + 1, j, k)
                v110 = vid(i + 1, j + 1, k)
                v010 = vid(i, j + 1, k)
                v001 = vid(i, j, k + 1)
                v101 = vid(i + 1, j, k + 1)
                v111 = vid(i + 1, j + 1, k + 1)
                v011 = vid(i, j + 1, k + 1)
                tets.extend(
                    [
                        [v000, v100, v110, v111],
                        [v000, v110, v010, v111],
                        [v000, v010, v011, v111],
                        [v000, v011, v001, v111],
                        [v000, v001, v101, v111],
                        [v000, v101, v100, v111],
                    ]
                )

    tets = np.asarray(tets, dtype=np.int32)
    return vertices, tets, _surface_faces_from_tets(tets)


def _bind_physics_material_any_prim(stage: Usd.Stage, prim_path: str, material_path: str) -> None:
    """Bind a physics material without requiring PhysX/Omni deformable APIs."""
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not prim.HasAPI(UsdShade.MaterialBindingAPI):
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    else:
        binding_api = UsdShade.MaterialBindingAPI(prim)
    binding_api.Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


@_clone
def spawn_newton_tet_cuboid(
    prim_path: str,
    cfg: NewtonTetCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a pre-tetrahedralized Newton cuboid deformable."""
    from pxr import UsdGeom

    from isaaclab.sim.utils import bind_visual_material, create_prim, get_current_stage

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    root_prim = create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    geom_path = f"{prim_path}/geometry"
    create_prim(geom_path, "Scope", stage=stage)

    vertices, tets, surface_faces = _cuboid_tet_grid(cfg.size, cfg.resolution)

    tet = UsdGeom.TetMesh.Define(stage, f"{geom_path}/sim_tet_mesh")
    tet.CreatePointsAttr(vertices)
    tet.CreateTetVertexIndicesAttr(tets.reshape(-1))
    tet.CreateSurfaceFaceVertexIndicesAttr(surface_faces.reshape(-1))

    mesh = UsdGeom.Mesh.Define(stage, f"{geom_path}/visual_mesh")
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexIndicesAttr(surface_faces.reshape(-1))
    mesh.CreateFaceVertexCountsAttr(np.full((surface_faces.shape[0],), 3, dtype=np.int32))
    mesh.CreateSubdivisionSchemeAttr("bilinear")

    if cfg.visual_material is not None and has_kit():
        material_path = (
            f"{geom_path}/{cfg.visual_material_path}"
            if not cfg.visual_material_path.startswith("/")
            else cfg.visual_material_path
        )
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(f"{geom_path}/visual_mesh", material_path, stage=stage)

    material_path = (
        f"{geom_path}/{cfg.physics_material_path}"
        if not cfg.physics_material_path.startswith("/")
        else cfg.physics_material_path
    )
    cfg.physics_material.func(material_path, cfg.physics_material)
    _bind_physics_material_any_prim(stage, prim_path, material_path)

    return root_prim


@_clone
def spawn_newton_vbd_tet_asset(
    prim_path: str,
    cfg: NewtonVbdTetAssetCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a legacy VBD tet asset as a standard Newton deformable TetMesh."""
    from pxr import UsdGeom

    from isaaclab.sim.utils import bind_visual_material, create_prim, get_current_stage

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    root_prim = create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    geom_path = f"{prim_path}/geometry"
    create_prim(geom_path, "Scope", stage=stage)

    vertices, tets, surface_faces = _vbd_tet_asset_geometry(
        cfg.usd_path,
        source_prim_path=cfg.source_prim_path,
        scale=cfg.scale,
        rotate_y_up_to_z_up=cfg.rotate_y_up_to_z_up,
        center_to_origin=cfg.center_to_origin,
    )

    tet = UsdGeom.TetMesh.Define(stage, f"{geom_path}/sim_tet_mesh")
    tet.CreatePointsAttr(vertices)
    tet.CreateTetVertexIndicesAttr(tets.reshape(-1))
    tet.CreateSurfaceFaceVertexIndicesAttr(surface_faces.reshape(-1))

    mesh = UsdGeom.Mesh.Define(stage, f"{geom_path}/visual_mesh")
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexIndicesAttr(surface_faces.reshape(-1))
    mesh.CreateFaceVertexCountsAttr(np.full((surface_faces.shape[0],), 3, dtype=np.int32))
    mesh.CreateSubdivisionSchemeAttr("bilinear")

    if cfg.visual_material is not None and has_kit():
        material_path = (
            f"{geom_path}/{cfg.visual_material_path}"
            if not cfg.visual_material_path.startswith("/")
            else cfg.visual_material_path
        )
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(f"{geom_path}/visual_mesh", material_path, stage=stage)

    material_path = (
        f"{geom_path}/{cfg.physics_material_path}"
        if not cfg.physics_material_path.startswith("/")
        else cfg.physics_material_path
    )
    cfg.physics_material.func(material_path, cfg.physics_material)
    _bind_physics_material_any_prim(stage, prim_path, material_path)

    return root_prim
