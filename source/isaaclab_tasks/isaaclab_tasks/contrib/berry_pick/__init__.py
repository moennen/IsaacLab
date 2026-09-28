# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous-grasp Franka teleoperation with deformable Gaussian berries."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Pick-Berry-Franka-IK-Rel-Newton",
    entry_point=f"{__name__}.env:BerryPickEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:BerryPickEnvCfg"},
)
