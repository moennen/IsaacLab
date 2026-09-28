# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local OVRTX daylight portals for the modeled greenhouse covering."""

import os

from isaaclab_tasks.contrib.gaussian_tasks.task_assets import task_asset_root


def configure_greenhouse_lighting(viewer):
    """Add diffuse transmitted daylight before the first RTX frame."""
    if os.environ.get("ISAACLAB_TOMATO_BACKGROUND", "modeled") not in {"1", "modeled"}:
        return
    if not hasattr(viewer, "add_background_usd") or getattr(viewer, "_tomato_daylight_added", False):
        return
    root = task_asset_root("tomato")
    viewer.add_background_usd(str(root / "greenhouse_modeled/lighting.usda"))
    viewer._tomato_daylight_added = True
