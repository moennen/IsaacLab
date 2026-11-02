# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera poses and lens settings shared by RTX and Warp berry views."""

import math

import numpy as np

VERTICAL_FOV_DEG = 45.0
VERTICAL_APERTURE_MM = 20.955
FOCAL_LENGTH_MM = VERTICAL_APERTURE_MM / (2 * math.tan(math.radians(VERTICAL_FOV_DEG) / 2))


def camera_view(view, berry):
    """Return the camera eye and look-at target in task coordinates, in metres."""
    if view == "room":
        return np.array([1.5, -2.0, 1.1]), np.array([0.25, 0.05, -0.18])
    if view == "punnet_and_bowl":
        return np.array([0.75, -0.24, 0.36]), np.array([0.46, 0.07, 0.035])
    if view == "berry_closeup":
        target = berry.offset + berry.positions().mean(0)
        return target + np.array([0.08, -0.025, 0.027]), target
    raise ValueError(f"Unknown camera view: {view}")
