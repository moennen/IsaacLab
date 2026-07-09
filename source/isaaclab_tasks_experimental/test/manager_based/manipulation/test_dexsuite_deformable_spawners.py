# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable import spawners
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.spawners import (
    _author_tet_material_arrays,
    _cuboid_tet_grid,
    _vbd_tet_asset_geometry,
    _vbd_tet_asset_material_arrays,
)


def test_cuboid_tet_grid_has_positive_volume_and_closed_surface():
    vertices, tets, surface_faces = _cuboid_tet_grid((0.09, 0.08, 0.07), (3, 3, 2))

    assert vertices.shape == (48, 3)
    assert tets.shape == (108, 4)
    assert surface_faces.shape == (84, 3)

    volumes = []
    for tet in tets:
        points = vertices[tet]
        volume = np.linalg.det(np.stack((points[1] - points[0], points[2] - points[0], points[3] - points[0]))) / 6.0
        volumes.append(volume)

    assert min(volumes) > 0.0
    assert np.isclose(sum(volumes), 0.09 * 0.08 * 0.07)

    edge_counts = Counter()
    for tri in surface_faces:
        for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_counts[tuple(sorted((int(edge[0]), int(edge[1]))))] += 1

    assert all(count == 2 for count in edge_counts.values())


def test_ragdoll_vbd_tet_asset_loads_with_positive_volume_and_closed_surface():
    asset_path = Path(spawners.__file__).parent / "assets" / "blueHairRagdoll100k_tet.usda"
    vertices, tets, surface_faces = _vbd_tet_asset_geometry(str(asset_path))

    assert vertices.shape == (255, 3)
    assert tets.shape == (673, 4)
    assert surface_faces.shape == (464, 3)
    np.testing.assert_allclose(vertices.mean(axis=0), np.zeros(3), atol=1.0e-7)
    np.testing.assert_allclose(np.ptp(vertices, axis=0), np.array([0.08524057, 0.33477156, 0.14859616]), atol=1.0e-6)

    volumes = []
    for tet in tets:
        points = vertices[tet]
        volume = np.linalg.det(np.stack((points[1] - points[0], points[2] - points[0], points[3] - points[0]))) / 6.0
        volumes.append(volume)

    assert min(volumes) > 0.0

    edge_counts = Counter()
    for tri in surface_faces:
        for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_counts[tuple(sorted((int(edge[0]), int(edge[1]))))] += 1

    assert all(count == 2 for count in edge_counts.values())


def test_vbd_tet_material_arrays_are_loaded_and_authored_on_simulation_tet_mesh(tmp_path):
    from pxr import Sdf, Usd, UsdGeom

    source_path = tmp_path / "source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source = source_stage.DefinePrim("/TetMesh", "Xform")
    source.CreateAttribute("newton:tetMu", Sdf.ValueTypeNames.FloatArray, custom=True).Set([1.0, 2.0])
    source.CreateAttribute("newton:tetLambda", Sdf.ValueTypeNames.FloatArray, custom=True).Set([3.0, 4.0])
    source.CreateAttribute("newton:particleMass", Sdf.ValueTypeNames.FloatArray, custom=True).Set([0.5, 0.6, 0.7, 0.8])
    source_stage.GetRootLayer().Save()

    arrays = _vbd_tet_asset_material_arrays(
        str(source_path),
        source_prim_path="/TetMesh",
        num_vertices=4,
        num_tets=2,
        scale=(2.0, 3.0, 4.0),
    )

    np.testing.assert_array_equal(arrays["newton:tetMu"], [1.0, 2.0])
    np.testing.assert_array_equal(arrays["newton:tetLambda"], [3.0, 4.0])
    np.testing.assert_allclose(arrays["newton:particleMass"], [12.0, 14.4, 16.8, 19.2])

    destination_stage = Usd.Stage.CreateInMemory()
    tet = UsdGeom.TetMesh.Define(destination_stage, "/World/sim_tet_mesh")
    _author_tet_material_arrays(tet, arrays)

    for name, expected in arrays.items():
        np.testing.assert_allclose(tet.GetPrim().GetAttribute(name).Get(), expected)
