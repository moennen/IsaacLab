# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local cache locations for the berry, tomato, and Gaussian-twin task assets."""

import os
from pathlib import Path


def task_asset_root(task: str, *, use_legacy_override: bool = True) -> Path:
    """Return a task's local asset directory, respecting legacy overrides.

    Download the Nucleus USD bundle with the adjacent ``assets.py`` CLI first.
    These standalone tasks use a verified local cache, not URL filesystem paths.
    Set ``use_legacy_override=False`` for shared resources such as backgrounds
    that must not follow a toy-package directory override.
    """
    if task not in {"berry", "tomato", "gaussian_twin"}:
        raise ValueError(f"Unknown task asset package: {task}")
    overrides = {"berry": "ISAACLAB_BERRY_ASSET_ROOT", "tomato": "ISAACLAB_TOMATO_ASSET_ROOT"}
    override = os.environ.get(overrides.get(task, "ISAACLAB_GAUSSIAN_TWIN_DIR")) if use_legacy_override else None
    root = os.environ.get("ISAACLAB_TASK_ASSET_ROOT", str(Path.home() / ".cache/isaaclab/tasks"))
    if "://" in (override or root):
        raise ValueError(
            "Task assets must be cached locally; run "
            "source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/assets.py --help."
        )
    value = override or str(Path(root) / task)
    return Path(value).expanduser()
