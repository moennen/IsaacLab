# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.tools.build_gaussian_vbd_asset_set import (  # noqa: E501
    _align_vomp_positions_to_tet_bounds,
    _scaled_proxy_vertices,
    _vomp_lookup_vertices,
    _vomp_youngs_modulus_correction_scale,
    _voxel_solid_from_gaussians,
    farthest_point_sample,
    filter_spawner_degenerate_tets,
    inset_tet_surface_for_collision_shell,
    material_arrays_from_vomp,
    tetrahedralize_proxy,
)


def test_farthest_point_sample_is_deterministic_and_has_requested_budget():
    points = np.asarray([[x, y, z] for x in range(3) for y in range(3) for z in range(3)], dtype=np.float32)

    first = farthest_point_sample(points, 8)
    second = farthest_point_sample(points, 8)

    assert first.shape == (8, 3)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first, axis=0)) == 8


def test_scaled_proxy_preserves_gaussian_mean_and_target_extent():
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    proxy, scale = _scaled_proxy_vertices(points, points, target_max_extent=0.4)

    np.testing.assert_allclose(proxy.mean(axis=0), points.mean(axis=0), atol=1.0e-7)
    assert np.isclose(scale, 0.2)
    assert np.isclose(np.ptp(proxy, axis=0).max(), 0.4)


def test_vomp_position_alignment_preserves_relative_material_coordinates():
    vomp_positions = np.asarray([[-0.5, -0.25, -0.5], [0.5, 0.25, 0.5]], dtype=np.float32)
    tet_vertices = np.asarray([[1.0, 2.0, 3.0], [1.4, 2.2, 3.4]], dtype=np.float32)

    aligned = _align_vomp_positions_to_tet_bounds(vomp_positions, tet_vertices)

    np.testing.assert_allclose(aligned, tet_vertices, atol=1.0e-7)


def test_vomp_lookup_vertices_rotates_default_y_up_tet_meshes_into_z_up():
    tet_vertices = np.asarray([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)

    lookup_vertices = _vomp_lookup_vertices(tet_vertices, source_y_up=True)

    np.testing.assert_array_equal(lookup_vertices, [[1.0, -3.0, 2.0], [-4.0, 6.0, 5.0]])
    np.testing.assert_array_equal(_vomp_lookup_vertices(tet_vertices, source_y_up=False), tet_vertices)


def test_vomp_material_mapping_uses_z_up_lookup_frame_for_y_up_sources(tmp_path):
    pytest.importorskip("scipy")
    voxels = np.zeros(
        8,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("youngs_modulus", "f4"),
            ("poisson_ratio", "f4"),
            ("density", "f4"),
        ],
    )
    grid = np.asarray([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    voxels["x"], voxels["y"], voxels["z"] = grid.T
    voxels["youngs_modulus"] = 10.0
    voxels["youngs_modulus"][np.all(grid == (-0.5, 0.5, -0.5), axis=1)] = 100.0
    voxels["poisson_ratio"] = 0.25
    voxels["density"] = 1.0
    material_path = tmp_path / "vomp.npz"
    np.savez(material_path, voxel_data=voxels)

    vertices = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]])
    tets = np.asarray([[0, 1, 2, 3]], dtype=np.int32)

    material_arrays = material_arrays_from_vomp(vertices, tets, material_path, source_y_up=True)

    np.testing.assert_allclose(material_arrays["newton:tetMu"], [40.0])
    np.testing.assert_allclose(material_arrays["newton:tetLambda"], [40.0])

    scaled_material_arrays = material_arrays_from_vomp(
        vertices, tets, material_path, source_y_up=True, youngs_modulus_scale=0.5
    )
    np.testing.assert_allclose(scaled_material_arrays["newton:tetMu"], [20.0])
    np.testing.assert_allclose(scaled_material_arrays["newton:tetLambda"], [20.0])


def test_vomp_youngs_modulus_correction_uses_one_global_mean_and_can_be_disabled(tmp_path):
    dtype = [("youngs_modulus", "f4")]
    first = np.zeros(2, dtype=dtype)
    second = np.zeros(2, dtype=dtype)
    first["youngs_modulus"] = [2.0, 4.0]
    second["youngs_modulus"] = [6.0, 8.0]
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    np.savez(first_path, voxel_data=first)
    np.savez(second_path, voxel_data=second)

    scale = _vomp_youngs_modulus_correction_scale([first_path, second_path], 1.0e5)

    assert scale == pytest.approx(2.0e4)
    assert _vomp_youngs_modulus_correction_scale([first_path, second_path], 1.0) == 1.0


def test_tetrahedralize_proxy_uses_all_vertices_when_scipy_is_available():
    pytest.importorskip("scipy")
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.2, 0.2, 0.2]],
        dtype=np.float32,
    )
    tets = tetrahedralize_proxy(vertices)

    assert tets.ndim == 2 and tets.shape[1] == 4
    assert tets.min() >= 0 and tets.max() < len(vertices)


def test_spawner_degeneracy_filter_removes_near_zero_cells_after_scaling():
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0e-9], [0.0, 0.0, 0.2]],
        dtype=np.float32,
    )
    tets = np.asarray([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int32)

    filtered = filter_spawner_degenerate_tets(vertices, tets)

    np.testing.assert_array_equal(filtered, np.asarray([[0, 1, 2, 4]], dtype=np.int32))


def test_collision_shell_insets_proxy_without_inverting_its_tet():
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    tets = np.asarray([[0, 1, 2, 3]], dtype=np.int32)

    inset, applied = inset_tet_surface_for_collision_shell(vertices, tets, collision_shell=0.05)

    assert applied > 0.0
    assert np.all(np.ptp(inset, axis=0) < np.ptp(vertices, axis=0))
    p = inset[tets][0]
    assert np.dot(p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0])) > 0.0


def test_collision_occupancy_considers_large_splats_beyond_nearest_centres():
    pytest.importorskip("scipy")
    positions = np.concatenate(
        (
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.repeat(np.asarray([[5.1, 0.0, 0.0]], dtype=np.float32), 32, axis=0),
        )
    )
    radii = np.concatenate((np.asarray([6.0], dtype=np.float32), np.full(32, 0.01, dtype=np.float32)))

    inner, origin, spacing, _ = _voxel_solid_from_gaussians(positions, radii, voxel_size=1.0, collision_shell=0.25)
    target_index = np.rint((np.asarray([5.0, 0.0, 0.0]) - origin) / spacing).astype(np.int64)

    assert inner[tuple(target_index)]
