# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka pad collision geometry shared by the tissue solvers, in meters."""

PAD_HALF_EXTENTS = (0.009, 0.0054, 0.0089)
PAD_OFFSETS = {"left": (0.0, 0.0054, 0.045), "right": (0.0, -0.0054, 0.045)}
