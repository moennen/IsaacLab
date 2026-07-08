# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.tools.build_gaussian_vbd_asset_set import (  # noqa: E501
    _scaled_proxy_vertices,
    _voxel_solid_from_gaussians,
    farthest_point_sample,
    filter_spawner_degenerate_tets,
    inset_tet_surface_for_collision_shell,
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
