# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton Gaussian storage follows scene-data deformable publications."""

from __future__ import annotations

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer

from isaaclab.scene_data.deformable_gaussian_splats import (
    GaussianSplatsEmbedding,
    GaussianSplatsPublication,
    GaussianSplatsState,
    WarpGaussianSplatsEmbedding,
    _sh_basis,
)
from isaaclab.test.utils.devices import test_devices


@pytest.mark.parametrize("device", test_devices())
def test_deformable_gaussian_splats_updates_native_data(device):
    """A world-space center reaches Newton's native array in shape-local coordinates."""
    builder = newton.ModelBuilder()
    coefficients = np.array([[0.2, 0.3, 0.4], [0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32)
    gaussian = newton.Gaussian(
        np.array([[0, 0, 0]], dtype=np.float32), sh_coeffs=coefficients.reshape(1, -1), sh_degree=1
    )
    builder.add_shape_gaussian(
        -1,
        gaussian=gaussian,
        xform=wp.transform((0, 0, 0), (0, 0, np.sqrt(0.5), np.sqrt(0.5))),
        scale=wp.vec3(2, 1, 1),
        label="/Splats",
    )
    second_gaussian = newton.Gaussian(
        np.array([[0, 0, 0]], dtype=np.float32),
        opacities=np.array([0.5]),
        sh_coeffs=coefficients.reshape(1, -1),
        sh_degree=1,
    )
    builder.add_shape_gaussian(-1, gaussian=second_gaussian, label="/Second")
    model = builder.finalize(device=device)
    publication = SimpleNamespace(
        geometry_timestamp=1,
        get_gaussian_splats_entries=lambda: [
            SimpleNamespace(gaussian_splats_embeddings=[SimpleNamespace(path="/Splats", sh_degree=0)])
        ],
    )
    visual = GaussianSplatsState(
        wp.array([[0, 20, 0], [30, 0, 0]], dtype=wp.vec3f, device=device),
        wp.array([[0, 0, 0, 1]] * 2, dtype=wp.quatf, device=device),
        wp.array([[1, 1, 1]] * 2, dtype=wp.vec3f, device=device),
        wp.array(coefficients, dtype=wp.vec3f, device=device),
    )
    binding = GaussianSplatsEmbedding(
        "/Splats",
        ("centers",),
        np.array([0, 1]),
        np.array([0]),
        np.array([1.0], dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.array([[0, 0, 0, 1]], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        coefficients,
        1,
        0,
        np.eye(4, dtype=np.float32),
    )
    warp_embedding = WarpGaussianSplatsEmbedding.create(
        binding, np.zeros((1, 3), dtype=np.float32), (0, 0, 0), (0, 0, 0, 1), device
    )
    states = GaussianSplatsPublication()
    states.batches.append((("/Splats", "/Second"), visual, warp_embedding))
    for index, path in enumerate(("/Splats", "/Second")):
        states[path] = GaussianSplatsState(
            visual.positions[index : index + 1],
            visual.orientations[index : index + 1],
            visual.scales[index : index + 1],
            visual.sh_coefficients,
            visual,
            index,
        )
    provider = SimpleNamespace(backend=publication, get_deformable_gaussian_splats=lambda: states)
    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)
    renderer.backend = SimpleNamespace(model=model, state_0=model.state())
    renderer._scene_data_provider = provider
    renderer._gaussian_splats_embeddings = {"/Splats": SimpleNamespace(sh_degree=0)}
    renderer._gaussian_splats_shapes = {"/Splats": 0, "/Second": 1}
    renderer._gaussian_splats_timestamp = -1
    renderer._gaussian_splats_targets = {}
    model.bvh_build_shapes(renderer.backend.state_0)

    renderer._update_deformable_gaussian_splats()
    np.testing.assert_allclose(gaussian._warp_data.transforms.numpy()[0, :3], [10, 0, 0], atol=1e-5)
    local_coefficients = gaussian._warp_data.sh_coeffs.numpy().reshape(-1, 3)
    np.testing.assert_allclose(local_coefficients[0], coefficients[0], atol=1e-6)
    np.testing.assert_allclose(
        _sh_basis(np.array([[0.0, -1.0, 0.0]]), 1) @ local_coefficients,
        _sh_basis(np.array([[1.0, 0.0, 0.0]]), 1) @ coefficients,
        atol=2e-4,
    )
    np.testing.assert_allclose(second_gaussian._warp_data.sh_coeffs.numpy().reshape(-1, 3), coefficients, atol=2e-4)
    assert model.bvh_shape_bounds.numpy()[0, 0, 0] > 5
    assert model.bvh_shapes.lowers.numpy()[0, 1] > 15
    np.testing.assert_allclose(second_gaussian._warp_data.transforms.numpy()[0, :3], [30, 0, 0], atol=1e-5)
    assert model.bvh_shape_bounds.numpy()[1, 0, 0] > 25
