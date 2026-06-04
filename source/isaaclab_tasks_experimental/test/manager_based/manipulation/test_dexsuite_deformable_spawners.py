# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import Counter

import numpy as np
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.spawners import _cuboid_tet_grid


def test_cuboid_tet_grid_has_positive_volume_and_closed_surface():
    vertices, tets, surface_faces = _cuboid_tet_grid((0.09, 0.08, 0.07), (3, 3, 2))

    assert vertices.shape == (48, 3)
    assert tets.shape == (108, 4)
    assert surface_faces.shape == (84, 3)

    volumes = []
    for tet in tets:
        points = vertices[tet]
        volume = np.linalg.det(
            np.stack((points[1] - points[0], points[2] - points[0], points[3] - points[0]))
        ) / 6.0
        volumes.append(volume)

    assert min(volumes) > 0.0
    assert np.isclose(sum(volumes), 0.09 * 0.08 * 0.07)

    edge_counts = Counter()
    for tri in surface_faces:
        for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_counts[tuple(sorted((int(edge[0]), int(edge[1]))))] += 1

    assert all(count == 2 for count in edge_counts.values())
