# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mutable Gaussian publications must remain spatially queryable and independent."""

from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from isaaclab_newton.renderers.gaussian_fields import NewtonGaussianFields

from isaaclab.scene_data import GaussianField, SceneDataProvider


@wp.kernel(enable_backward=False)
def _query_fields(
    fields: wp.array(dtype=newton.Gaussian.Data),
    indices: wp.array(dtype=wp.uint64),
    centers: wp.array(dtype=wp.vec3f),
    hits: wp.array(dtype=int),
    alphas: wp.array(dtype=float),
):
    i = wp.tid()
    data = fields[int(indices[i])]
    query = wp.bvh_query_aabb(data.bvh_id, centers[i] - wp.vec3f(0.001), centers[i] + wp.vec3f(0.001))
    index = int(0)
    while wp.bvh_query_next(query, index):
        hits[i] += 1
    alphas[i] = data.opacities[0]


def test_deduplicated_gaussian_shapes_deform_independently_and_refit_bounds():
    """Moving two identical assets beyond rest bounds updates both BVHs without altering physics data."""
    builder = newton.ModelBuilder()
    gaussian = newton.Gaussian(np.array([[0, 0, 0]], np.float32), scales=np.full((1, 3), 0.05), opacities=[0.7])
    builder.add_shape_gaussian(-1, gaussian=gaussian, label="/Left")
    builder.add_shape_gaussian(-1, gaussian=gaussian, label="/Right")
    model = builder.finalize(device="cpu")
    backend = SimpleNamespace(model=model, state_0=model.state())
    versions = SimpleNamespace(transforms_timestamp=0, geometry_timestamp=0)
    provider = SceneDataProvider(versions)
    for path, center in (("/Left", [2, 0, 0]), ("/Right", [-2, 0, 0])):
        provider.register_gaussian_field(
            path,
            GaussianField(
                wp.array([center], dtype=wp.vec3f, device="cpu"),
                wp.array([[0, 0, 0, 1]], dtype=wp.quatf, device="cpu"),
                wp.full(1, wp.vec3f(0.1), device="cpu"),
                opacities=wp.array([0.3], dtype=float, device="cpu") if path == "/Left" else None,
            ),
        )
    writer = NewtonGaussianFields(backend, provider)
    writer.update()
    centers = wp.array([[2, 0, 0], [-2, 0, 0]], dtype=wp.vec3f, device="cpu")
    hits = wp.zeros(2, dtype=int, device="cpu")
    alphas = wp.empty(2, dtype=float, device="cpu")
    wp.launch(
        _query_fields,
        2,
        [writer.model.gaussians_data, writer.model.shape_source_ptr, centers, hits, alphas],
        device="cpu",
    )
    np.testing.assert_array_equal(hits.numpy(), [1, 1])
    np.testing.assert_allclose(alphas.numpy(), [0.3, 0.7])
    bounds = writer.model.bvh_shape_bounds.numpy()
    assert bounds[0, 0, 0] > 1 and bounds[1, 1, 0] < -1
    hits.zero_()
    wp.launch(_query_fields, 2, [model.gaussians_data, model.shape_source_ptr, centers, hits, alphas], device="cpu")
    np.testing.assert_array_equal(hits.numpy(), [0, 0])
    provider.get_gaussian_fields()["/Left"].positions.assign([[4, 0, 0]])
    versions.geometry_timestamp += 1
    writer.update()
    centers.assign([[4, 0, 0], [-2, 0, 0]])
    hits.zero_()
    wp.launch(
        _query_fields,
        2,
        [writer.model.gaussians_data, writer.model.shape_source_ptr, centers, hits, alphas],
        device="cpu",
    )
    np.testing.assert_array_equal(hits.numpy(), [1, 1])
    provider.unregister_gaussian_field("/Left")
    writer.update()
    centers.assign([[0, 0, 0], [-2, 0, 0]])
    hits.zero_()
    wp.launch(
        _query_fields,
        2,
        [writer.model.gaussians_data, writer.model.shape_source_ptr, centers, hits, alphas],
        device="cpu",
    )
    np.testing.assert_array_equal(hits.numpy(), [1, 1])
    np.testing.assert_allclose(alphas.numpy(), [0.7, 0.7])
