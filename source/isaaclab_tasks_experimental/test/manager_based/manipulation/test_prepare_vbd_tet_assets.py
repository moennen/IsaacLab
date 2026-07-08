# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.tools.prepare_vbd_tet_assets import (
    _load_vbd_tet_asset,
    prepare_vbd_tet_assets,
    write_vbd_tet_asset,
)


def _tet() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]], dtype=np.float32),
        np.array([[0, 1, 2, 3]], dtype=np.int32),
    )


def test_prepare_vbd_tet_assets_normalizes_and_writes_common_topology(tmp_path):
    vertices, tets = _tet()
    first = tmp_path / "first.usda"
    second = tmp_path / "second.usda"
    write_vbd_tet_asset(first, vertices, tets)
    write_vbd_tet_asset(second, vertices + np.array([4.0, -2.0, 1.0], dtype=np.float32), tets)

    prepared = prepare_vbd_tet_assets(
        [first, second], tmp_path / "prepared", target_num_vertices=4, target_num_tets=1, target_max_extent=0.4
    )

    assert len(prepared) == 2
    assert all(item.num_vertices == 4 and item.num_tets == 1 for item in prepared)
    for item in prepared:
        output_vertices, output_tets = _load_vbd_tet_asset(item.path)
        np.testing.assert_allclose(output_vertices.mean(axis=0), np.zeros(3), atol=1.0e-7)
        assert np.isclose(np.ptp(output_vertices, axis=0).max(), 0.4)
        assert output_tets.shape == (1, 4)


def test_prepare_vbd_tet_assets_rejects_a_mismatched_budget(tmp_path):
    vertices, tets = _tet()
    source = tmp_path / "source.usda"
    write_vbd_tet_asset(source, vertices, tets)

    with pytest.raises(ValueError, match="common topology budget"):
        prepare_vbd_tet_assets([source], tmp_path / "prepared", target_num_vertices=5, target_num_tets=1)
