# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD task metadata round trips without JSON, NPZ, or pickle sidecars."""

import numpy as np

from pxr import Usd

from isaaclab_tasks.contrib.gaussian_tasks.usd_task_data import read_array, read_data, write_array, write_data


def test_nested_metadata_and_numeric_arrays_roundtrip(tmp_path):
    path = tmp_path / "data.usdc"
    stage = Usd.Stage.CreateNew(str(path))
    value = {"name": "raspberry", "nested": [None, True, 42, 1.5, {}, []], "path:with spaces": "attribution"}
    write_data(stage, "/Data", value)
    arrays = {
        "scalar": np.array(0.001, np.float32),
        "empty": np.zeros((0, 3), np.int32),
        "indices": np.array([0, -1, 2**35], np.int64),
        "matrix": np.eye(3, dtype=np.float32),
    }
    for key, array in arrays.items():
        write_array(stage.GetPrimAtPath("/Data"), key, array)
    stage.GetRootLayer().Save()
    restored = Usd.Stage.Open(str(path))
    assert read_data(restored, "/Data") == value
    for key, array in arrays.items():
        result = read_array(restored.GetPrimAtPath("/Data"), key)
        np.testing.assert_array_equal(result, array)
        assert result.dtype == array.dtype
        assert result.shape == array.shape
