# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX writes for shared, local-space Gaussian publications."""

import contextlib

import numpy as np
import ovstage
import warp as wp
from ovrtx import DataAccess, Semantic

from pxr import UsdGeom

from isaaclab.sim import get_current_stage


@wp.kernel(enable_backward=False)
def _init_bounds(bounds: wp.array2d(dtype=float)):
    axis = wp.tid()
    bounds[0, axis] = wp.inf
    bounds[1, axis] = -wp.inf


@wp.kernel(enable_backward=False)
def _bounds(positions: wp.array(dtype=wp.vec3f), scales: wp.array(dtype=wp.vec3f), bounds: wp.array2d(dtype=float)):
    i = wp.tid()
    radius = 4.0 * wp.max(scales[i][0], wp.max(scales[i][1], scales[i][2]))
    for axis in range(3):
        wp.atomic_min(bounds, 0, axis, positions[i][axis] - radius)
        wp.atomic_max(bounds, 1, axis, positions[i][axis] + radius)


@wp.kernel(enable_backward=False)
def _normalize(
    positions: wp.array(dtype=wp.vec3f),
    scales: wp.array(dtype=wp.vec3f),
    center: wp.vec3f,
    rest_center: wp.vec3f,
    factor: float,
    output_positions: wp.array(dtype=wp.vec3f),
    output_scales: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    output_positions[i] = (positions[i] - center) / factor + rest_center
    output_scales[i] = scales[i] / factor


class OVRTXGaussianFields:
    """Own Gaussian destination resources independently of the simulation producer.

    OVRTX retains the authored field extent. Normalize changing arrays into that extent
    and compensate with a local transform, so translation and stretching cannot be culled
    against stale bounds. The small bounds readback is required by the native scene writer.
    """

    def __init__(self, renderer):
        self.renderer = renderer
        self._fields = {}
        self._timestamp = None
        self._queries = {}

    def update(self):
        """Publish each changed simulation state into the renderer's detached scene."""
        provider = self.renderer._sdp
        fields = provider.get_gaussian_fields()
        if self._timestamp == provider.gaussian_timestamp:
            return
        for path in self._fields.keys() - fields.keys():
            self._restore(path, self._fields.pop(path))
            resources = self._queries.pop(path, None)
            if resources is not None:
                resources.close()
        for path, field in fields.items():
            cached = self._fields.get(path)
            if cached is None or cached[0] is not field:
                if cached is not None:
                    self._restore(path, cached)
                prim = get_current_stage().GetPrimAtPath(path)
                if len(prim.GetAttribute("positions").Get()) != len(field.positions):
                    raise ValueError(f"Gaussian count must match the authored field at {path}.")
                extent = np.asarray(prim.GetAttribute("extent").Get(), dtype=np.float64)
                if extent.shape != (2, 3) or np.any(extent[1] <= extent[0]):
                    raise ValueError(f"A mutable Gaussian field requires a nonempty authored extent: {path}.")
                local = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation(0))
                query = None
                if self.renderer._use_ovstage:
                    resources = self._queries.pop(path, None)
                    if resources is not None:
                        resources.close()
                    self._queries[path] = resources = contextlib.ExitStack()
                    query = resources.enter_context(self.renderer.scene.query([path]))
                with wp.ScopedDevice(field.positions.device):
                    cached = (
                        field,
                        extent.mean(0),
                        (extent[1] - extent[0]) / 2,
                        local,
                        query,
                        wp.empty((2, 3), dtype=float),
                        wp.empty_like(field.positions),
                        wp.empty_like(field.scales),
                    )
                self._fields[path] = cached
            _, rest_center, rest_half, local, query, bounds, positions, scales = cached
            with wp.ScopedDevice(field.positions.device):
                wp.launch(_init_bounds, 3, [bounds])
                wp.launch(_bounds, len(field.positions), [field.positions, field.scales, bounds])
                extent = bounds.numpy().astype(np.float64)
                center = extent.mean(0)
                factor = float(np.max((extent[1] - extent[0]) / (2 * rest_half)))
                if not np.isfinite(factor) or factor <= 0:
                    raise ValueError(f"Invalid deformed Gaussian bounds at {path}.")
                wp.launch(
                    _normalize,
                    len(field.positions),
                    [
                        field.positions,
                        field.scales,
                        wp.vec3f(*center),
                        wp.vec3f(*rest_center),
                        factor,
                        positions,
                        scales,
                    ],
                )
                transform = np.eye(4)
                transform[:3, :3] *= factor
                transform[3, :3] = center - factor * rest_center
                self._write_transform(path, query, transform @ local)
                arrays = {"positions": positions, "scales": scales, "orientations": field.orientations}
                if field.opacities is not None:
                    arrays["opacities"] = field.opacities
                arrays.update(field.attributes)
                for name, array in arrays.items():
                    self._write_array(path, query, name, array)
        self._timestamp = provider.gaussian_timestamp

    def _restore(self, path, cached):
        field, _center, _half, local, query, *_buffers = cached
        prim = get_current_stage().GetPrimAtPath(path)
        self._write_transform(path, query, local)
        arrays = {"positions": field.positions, "orientations": field.orientations, "scales": field.scales}
        if field.opacities is not None:
            arrays["opacities"] = field.opacities
        arrays.update(field.attributes)
        for name, source in arrays.items():
            values = np.asarray(prim.GetAttribute(name).Get(), dtype=np.float32)
            restored = wp.array(values, dtype=source.dtype, device=source.device)
            self._write_array(path, query, name, restored)

    def _write_transform(self, path, query, transform):
        renderer = self.renderer
        reset = UsdGeom.Xformable(get_current_stage().GetPrimAtPath(path)).GetResetXformStack()
        if renderer._use_ovstage:
            from .ovrtx_renderer import xform_tensor_from_numpy

            for name, value, semantic in (
                ("omni:resetXformStack", np.array([reset], dtype=np.bool_), ovstage.AttributeSemantic.NONE),
                ("omni:xform", xform_tensor_from_numpy(transform[None]), ovstage.AttributeSemantic.MATRIX),
            ):
                renderer.scene.stage.write_attribute(
                    query,
                    name,
                    ordinal=renderer.scene.ordinal,
                    tensors=value,
                    is_array=False,
                    semantic=semantic,
                ).wait()
        else:
            renderer.backend.renderer.write_attribute([path], "omni:resetXformStack", np.array([reset], dtype=np.bool_))
            renderer.backend.renderer.write_attribute(
                [path],
                "omni:xform",
                transform[None],
                semantic=Semantic.XFORM_MAT4x4,
            )

    def _write_array(self, path, query, name, array):
        renderer = self.renderer
        stream = wp.get_stream(array.device).cuda_stream or 1
        if renderer._use_ovstage:
            tensor = ovstage.make_dltensor(
                array,
                dtype=ovstage.numpy_to_dldatatype(
                    np.dtype(np.float32), lanes=wp.types.type_size_in_bytes(array.dtype) // 4
                ),
                shape=[len(array)],
            )
            renderer.scene.stage.write_attribute(
                query,
                name,
                ordinal=renderer.scene.ordinal,
                tensors=[tensor],
                is_array=True,
                cuda_stream=stream,
            ).wait()
        else:
            renderer.backend.renderer.write_array_attribute(
                [path],
                name,
                [array],
                data_access=DataAccess.ASYNC,
                cuda_stream=stream,
            )

    def close(self):
        """Release native queries after pending renders have completed."""
        for resources in self._queries.values():
            resources.close()
        self._queries.clear()
        self._fields.clear()
