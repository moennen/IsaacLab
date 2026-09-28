# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single workcell environment with resettable tissue, damage, and contact history."""

from isaaclab.envs import ManagerBasedRLEnv

from .runtime import BerryRuntime


class BerryPickEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.scene.num_envs != 1:
            raise ValueError("Berry teleoperation currently supports one environment")
        super().__init__(cfg, **kwargs)
        self.berry = BerryRuntime(cfg, self.scene["robot"])

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "berry"):
            self.berry.reset(self.scene["robot"])

    def step(self, action):
        result = super().step(action)
        self.berry.checker.check()
        return result
