# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka IK teleoperation for a deformable, force-detachable tomato plant."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Pick-Tomato-Franka-IK-Rel-Newton",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:TomatoPickEnvCfg"},
)
