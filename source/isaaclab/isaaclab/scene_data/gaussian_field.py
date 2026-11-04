# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer-independent publication of mutable Gaussian fields."""

from collections.abc import Callable
from dataclasses import dataclass, field

import warp as wp


@dataclass
class GaussianField:
    """Borrowed, fixed-topology Gaussian buffers in the authored prim's local frame.

    Producers own the arrays and allocate them before simulation graph capture. Renderers
    borrow them read-only until the next update. Positions and scales are in meters;
    orientations use Warp's xyzw quaternion order. Optional attributes are float scalar
    or vector USD arrays (for example shader primvars). Authored appearance is preserved
    when opacities or attributes are omitted. Replacing buffers requires unregistering
    and registering the field again, after pending renders have completed.

    Args:
        positions: Gaussian centers, ``wp.vec3f[count]``.
        orientations: Gaussian rotations, ``wp.quatf[count]``.
        scales: Gaussian standard deviations, ``wp.vec3f[count]``.
        opacities: Optional opacity array, ``wp.float32[count]``.
        attributes: Additional per-Gaussian float attributes by USD attribute name.
        update: Optional producer refresh, called once per simulation state before use.
    """

    positions: wp.array
    orientations: wp.array
    scales: wp.array
    opacities: wp.array | None = None
    attributes: dict[str, wp.array] = field(default_factory=dict)
    update: Callable[[], None] | None = None
