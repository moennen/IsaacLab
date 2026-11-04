# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Newton destinations for shared mutable Gaussian fields."""

import copy

import newton
import warp as wp


@wp.kernel(enable_backward=False)
def _copy_gaussian_data(
    source: wp.array(dtype=newton.Gaussian.Data),
    indices: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=newton.Gaussian.Data),
):
    i = wp.tid()
    destination[i] = source[indices[i]]


@wp.kernel(enable_backward=False)
def _restore_field(
    source: wp.array(dtype=newton.Gaussian.Data),
    source_index: int,
    destination: wp.array(dtype=newton.Gaussian.Data),
    destination_index: int,
    rest_bounds: wp.array2d(dtype=wp.vec3f),
    bounds: wp.array2d(dtype=wp.vec3f),
    shape: int,
):
    destination[destination_index] = source[source_index]
    bounds[shape, 0] = rest_bounds[shape, 0]
    bounds[shape, 1] = rest_bounds[shape, 1]


@wp.kernel(enable_backward=False)
def _copy_opacities(data_array: wp.array(dtype=newton.Gaussian.Data), index: int, opacities: wp.array(dtype=float)):
    i = wp.tid()
    data = data_array[index]
    opacities[i] = data.opacities[i]


@wp.kernel(enable_backward=False)
def _bind_field(
    data_array: wp.array(dtype=newton.Gaussian.Data),
    index: int,
    transforms: wp.array(dtype=wp.transformf),
    scales: wp.array(dtype=wp.vec3f),
    opacities: wp.array(dtype=float),
    bvh_id: wp.uint64,
):
    data = data_array[index]
    data.transforms = transforms
    data.scales = scales
    data.opacities = opacities
    data.bvh_id = bvh_id
    data_array[index] = data


@wp.kernel(enable_backward=False)
def _write_gaussians(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    scales: wp.array(dtype=wp.vec3f),
    gaussians: wp.array(dtype=newton.Gaussian.Data),
    gaussian_index: int,
    lowers: wp.array(dtype=wp.vec3f),
    uppers: wp.array(dtype=wp.vec3f),
    bounds: wp.array2d(dtype=float),
):
    i = wp.tid()
    data = gaussians[gaussian_index]
    data.transforms[i] = wp.transformf(positions[i], orientations[i])
    data.scales[i] = scales[i]
    # A sphere encloses the rotated ellipsoid at the native response cutoff.
    cutoff = wp.sqrt(-2.0 * wp.log(data.min_response))
    radius = cutoff * wp.max(scales[i][0], wp.max(scales[i][1], scales[i][2]))
    lowers[i] = positions[i] - wp.vec3f(radius)
    uppers[i] = positions[i] + wp.vec3f(radius)
    for axis in range(3):
        wp.atomic_min(bounds, 0, axis, lowers[i][axis])
        wp.atomic_max(bounds, 1, axis, uppers[i][axis])


@wp.kernel(enable_backward=False)
def _write_opacities(opacities: wp.array(dtype=float), gaussians: wp.array(dtype=newton.Gaussian.Data), index: int):
    i = wp.tid()
    data = gaussians[index]
    data.opacities[i] = opacities[i]


@wp.kernel(enable_backward=False)
def _init_bounds(bounds: wp.array2d(dtype=float)):
    axis = wp.tid()
    bounds[0, axis] = wp.inf
    bounds[1, axis] = -wp.inf


@wp.kernel(enable_backward=False)
def _write_shape_bounds(bounds: wp.array2d(dtype=float), shape: int, output: wp.array2d(dtype=wp.vec3f)):
    output[shape, 0] = wp.vec3f(bounds[0, 0], bounds[0, 1], bounds[0, 2])
    output[shape, 1] = wp.vec3f(bounds[1, 0], bounds[1, 1], bounds[1, 2])


class NewtonGaussianFields:
    """Write publications into finalized model buffers and refit their native BVHs."""

    def __init__(self, backend, provider):
        self.backend = backend
        self.provider = provider
        # A small render-only model view gives every Gaussian shape its own data slot.
        # Static arrays stay borrowed. Mutable publications receive private buffers on first use,
        # even when the importer deduplicated several shapes into one Gaussian asset.
        self.model = model = copy.copy(backend.model)
        self._shapes = {path: i for i, path in enumerate(model.shape_label)}
        indices = model.shape_source_ptr.numpy().copy()
        self._original_indices = indices.copy()
        self._rest_bounds = wp.clone(model.bvh_shape_bounds)
        shapes = [i for i, source in enumerate(model.shape_source) if isinstance(source, newton.Gaussian)]
        model.gaussians_data = wp.empty(len(shapes), dtype=newton.Gaussian.Data, device=model.device)
        if shapes:
            source_indices = wp.array(indices[shapes], dtype=wp.int32, device=model.device)
            wp.launch(
                _copy_gaussian_data,
                len(shapes),
                [backend.model.gaussians_data, source_indices, model.gaussians_data],
                device=model.device,
            )
        for index, shape in enumerate(shapes):
            indices[shape] = index
        model.gaussians_count = len(shapes)
        model.shape_source_ptr = wp.array(indices, dtype=model.shape_source_ptr.dtype, device=model.device)
        self._indices = indices
        self._bindings = {}
        self._timestamp = None

    def update(self):
        """Refresh changed fields, preserving all finalized buffer addresses for captured queries."""
        fields = self.provider.get_gaussian_fields()
        timestamp = self.provider.gaussian_timestamp
        if self._timestamp == timestamp:
            return
        model = self.model
        removed = self._bindings.keys() - fields.keys()
        for path in removed:
            _, shape, index, *_resources = self._bindings.pop(path)
            wp.launch(
                _restore_field,
                1,
                [
                    self.backend.model.gaussians_data,
                    int(self._original_indices[shape]),
                    model.gaussians_data,
                    index,
                    self._rest_bounds,
                    model.bvh_shape_bounds,
                    shape,
                ],
                device=model.device,
            )
        for path, field in fields.items():
            cached = self._bindings.get(path)
            if cached is None or cached[0] is not field:
                shape = self._shapes[path]
                gaussian = model.shape_source[shape]
                if not isinstance(gaussian, newton.Gaussian):
                    raise ValueError(f"Published Gaussian path is not a native Gaussian shape: {path}.")
                index = int(self._indices[shape])
                if gaussian.count != len(field.positions) or model.device != field.positions.device:
                    raise ValueError(f"Gaussian count and device must match the imported field at {path}.")
                with wp.ScopedDevice(model.device):
                    transforms = wp.empty(len(field.positions), dtype=wp.transformf)
                    scales = wp.empty_like(field.scales)
                    opacities = wp.empty(len(field.positions), dtype=float)
                    bvh = wp.Bvh(wp.clone(field.positions), wp.clone(field.positions))
                    wp.launch(
                        _copy_opacities,
                        len(field.positions),
                        [self.backend.model.gaussians_data, int(self._original_indices[shape]), opacities],
                    )
                    wp.launch(_bind_field, 1, [model.gaussians_data, index, transforms, scales, opacities, bvh.id])
                    cached = (field, shape, index, bvh, transforms, scales, opacities, wp.empty((2, 3), dtype=float))
                self._bindings[path] = cached
            _, shape, index, bvh, _transforms, _scales, _opacities, bounds = cached
            with wp.ScopedDevice(model.device):
                wp.launch(_init_bounds, 3, [bounds])
                wp.launch(
                    _write_gaussians,
                    len(field.positions),
                    [
                        field.positions,
                        field.orientations,
                        field.scales,
                        model.gaussians_data,
                        index,
                        bvh.lowers,
                        bvh.uppers,
                        bounds,
                    ],
                )
                if field.opacities is not None:
                    wp.launch(_write_opacities, len(field.positions), [field.opacities, model.gaussians_data, index])
                bvh.refit()
                wp.launch(_write_shape_bounds, 1, [bounds, shape, model.bvh_shape_bounds])
        if fields or removed:
            model.bvh_refit_shapes(self.backend.state_0)
        self._timestamp = timestamp
