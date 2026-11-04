# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gaussian linear-blend skinning from the optional Kaolin tissue's baked handles."""

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from .mpm_binding import gaussian_shape


@wp.kernel(enable_backward=False)
def _skin_gaussians(
    rest: wp.array(dtype=wp.vec3f),
    weights: wp.array2d(dtype=float),
    transforms: wp.array(dtype=wp.mat44f),
    placement_rotation: wp.mat33f,
    world_offset: wp.vec3f,
    base: wp.array(dtype=wp.mat33f),
    positions: wp.array(dtype=wp.vec3f),
    gradient: wp.array(dtype=wp.mat33f),
    covariance: wp.array(dtype=wp.mat33f),
):
    i = wp.tid()
    f = wp.identity(n=3, dtype=float)
    translation = wp.vec3f(0.0)
    for handle in range(weights.shape[1]):
        t = transforms[handle]
        rotation = wp.mat33f(t[0, 0], t[0, 1], t[0, 2], t[1, 0], t[1, 1], t[1, 2], t[2, 0], t[2, 1], t[2, 2])
        f += weights[i, handle] * rotation
        translation += weights[i, handle] * wp.vec3f(t[0, 3], t[1, 3], t[2, 3])
    positions[i] = f @ rest[i] + translation - world_offset
    local_f = f @ wp.transpose(placement_rotation)
    gradient[i] = local_f
    covariance[i] = local_f @ base[i]


class SimplicitsBinding:
    """Use the asset's exact Gaussian skinning weights with Kaolin's public handle transforms."""

    def __init__(self, tissue):
        self.tissue = tissue
        data = tissue.particles["simplicits"]
        count = len(tissue.gaussians["xyz"])
        if len(data["renderable_weights"]) != count:
            raise ValueError("Baked Simplicits skinning must match every Gaussian in the asset")
        rotations = Rotation.from_quat(tissue.gaussians["rotations"]).as_matrix()
        base = rotations * tissue.gaussians["scales"][:, None, :]
        self._placement = wp.mat33f(tissue.particles["simplicits_transform"][:3, :3])
        self._offset = wp.vec3f(*tissue.offset)
        with wp.ScopedDevice(tissue.device):
            self._rest = wp.array(data["renderable_pts"], dtype=wp.vec3f)
            self._weights = wp.array(data["renderable_weights"], dtype=float)
            self._base = wp.array(base.astype(np.float32), dtype=wp.mat33f)
            self.positions = wp.empty(count, dtype=wp.vec3f)
            self.scales = wp.empty(count, dtype=wp.vec3f)
            self.orientations = wp.empty(count, dtype=wp.quatf)
            self.shading = wp.empty(count, dtype=wp.vec4f)
            self._gradient = wp.empty(count, dtype=wp.mat33f)
            self._covariance = wp.empty(count, dtype=wp.mat33f)
            self._nearest = wp.zeros(count, dtype=int)
            self._damage = wp.zeros(1, dtype=float)

    def deform(self, particles, damage):
        """Publish native handle deformation, covariance and SH rotation; Simplicits has no tissue-damage field."""
        transforms = self.tissue.simplicits_solver.scene.get_object_transforms(
            self.tissue.simplicits_object
        ).contiguous()
        with wp.ScopedDevice(self.tissue.device):
            wp.launch(
                _skin_gaussians,
                len(self.positions),
                [
                    self._rest,
                    self._weights,
                    wp.from_torch(transforms, dtype=wp.mat44f),
                    self._placement,
                    self._offset,
                    self._base,
                    self.positions,
                    self._gradient,
                    self._covariance,
                ],
            )
            wp.launch(
                gaussian_shape,
                len(self.positions),
                [
                    self._covariance,
                    self._gradient,
                    self._nearest,
                    self._damage,
                    0.0,
                    self.scales,
                    self.orientations,
                    self.shading,
                ],
            )
