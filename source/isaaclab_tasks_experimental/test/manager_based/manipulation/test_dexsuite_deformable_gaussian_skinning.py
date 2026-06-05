# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.tools.package_skinned_gaussian_tet_asset import (  # noqa: E501
    compute_tet_barycentric_skinning,
    package_skinned_gaussian_tet_asset,
    reconstruct_points_from_skinning,
)

from pxr import Gf, Sdf, Usd, UsdGeom


def test_compute_tet_barycentric_skinning_reconstructs_points():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    tets = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    query = np.asarray(
        [
            [0.25, 0.25, 0.25],
            [1.25, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = compute_tet_barycentric_skinning(query, vertices, tets)

    np.testing.assert_array_equal(result.tet_indices, np.asarray([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(result.influence_indices, np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32))
    np.testing.assert_allclose(result.influence_weights[0], np.asarray([0.25, 0.25, 0.25, 0.25]), atol=1.0e-7)
    np.testing.assert_allclose(result.influence_weights[1], np.asarray([-0.25, 1.25, 0.0, 0.0]), atol=1.0e-7)
    assert result.barycentric_violation[0] == 0.0
    assert result.barycentric_violation[1] > 0.0
    np.testing.assert_allclose(reconstruct_points_from_skinning(vertices, result), query, atol=1.0e-7)


def test_package_skinned_gaussian_tet_asset_writes_custom_binding(tmp_path):
    gaussian_path = tmp_path / "gaussian.usda"
    tet_path = tmp_path / "tet.usda"
    output_path = tmp_path / "combined.usda"

    gaussian_stage = Usd.Stage.CreateNew(str(gaussian_path))
    gaussian_prim = gaussian_stage.DefinePrim("/World/Gaussians/gaussians_0", "ParticleField3DGaussianSplat")
    gaussian_prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set(
        [Gf.Vec3f(0.25, 0.25, 0.25), Gf.Vec3f(0.50, 0.25, 0.125)]
    )
    gaussian_prim.CreateAttribute("opacities", Sdf.ValueTypeNames.FloatArray).Set([1.0, 0.5])
    gaussian_stage.Save()

    tet_stage = Usd.Stage.CreateNew(str(tet_path))
    tet_prim = tet_stage.DefinePrim("/TetMesh", "Xform")
    tet_prim.CreateAttribute("vbd:vertices", Sdf.ValueTypeNames.Point3fArray).Set(
        [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(0.0, 0.0, 1.0)]
    )
    tet_prim.CreateAttribute("vbd:tet_indices", Sdf.ValueTypeNames.IntArray).Set([0, 1, 2, 3])
    tet_stage.Save()

    package_skinned_gaussian_tet_asset(
        gaussian_usd_path=str(gaussian_path),
        tet_usd_path=str(tet_path),
        output_usd_path=str(output_path),
        rotate_tet_y_up_to_z_up=False,
        center_tet_to_origin=False,
        chunk_size=16,
    )

    stage = Usd.Stage.Open(str(output_path))
    gaussian = stage.GetPrimAtPath("/SkinnedGaussianTetAsset/visual/gaussians_0")
    tet = UsdGeom.TetMesh(stage.GetPrimAtPath("/SkinnedGaussianTetAsset/physics/sim_tet_mesh"))

    assert gaussian.IsValid()
    assert tet
    assert gaussian.GetRelationship("newton:deformableSkin:targetTetMesh").GetTargets() == [tet.GetPath()]
    assert gaussian.GetAttribute("newton:deformableSkin:schemaVersion").Get() == 1
    assert gaussian.GetAttribute("newton:deformableSkin:influenceSize").Get() == 4
    assert gaussian.GetAttribute("newton:deformableSkin:pointCount").Get() == 2
    assert gaussian.GetAttribute("newton:deformableSkin:influenceIndices").Get() == [0, 1, 2, 3, 0, 1, 2, 3]
    np.testing.assert_allclose(
        np.asarray(gaussian.GetAttribute("newton:deformableSkin:influenceWeights").Get()).reshape(-1, 4),
        np.asarray([[0.25, 0.25, 0.25, 0.25], [0.125, 0.5, 0.25, 0.125]], dtype=np.float32),
        atol=1.0e-7,
    )
    assert len(tet.GetPrim().GetAttribute("newton:deformableSkin:restTetBasisInv").Get()) == 9
