# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless Newton deformable spawners for the experimental Dexsuite task."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import MISSING

import numpy as np

from pxr import Usd, UsdGeom, UsdShade

from isaaclab.sim.spawners.materials.physics_materials_cfg import DeformableBodyMaterialBaseCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import VisualMaterialCfg
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg
from isaaclab.sim.utils import bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils.configclass import configclass
from isaaclab.utils.version import has_kit


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

    surface = [oriented_faces[face] for face, count in face_counts.items() if count == 1]

    return vertices, tets, np.asarray(surface, dtype=np.int32)


def _bind_physics_material_any_prim(stage: Usd.Stage, prim_path: str, material_path: str) -> None:
    """Bind a physics material without requiring PhysX/Omni deformable APIs."""
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


@clone
def spawn_newton_tet_cuboid(
    prim_path: str,
    cfg: NewtonTetCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a pre-tetrahedralized Newton cuboid deformable."""
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
