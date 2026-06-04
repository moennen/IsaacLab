# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum helpers for the deformable Dexsuite task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def initial_final_interpolate_fn(
    env: ManagerBasedRLEnv,
    env_id,
    data,
    initial_value,
    final_value,
    difficulty_term_str,
):
    """Interpolate nested values using the named difficulty term's normalized progress."""
    difficulty_term: DeformableCommandDifficultyScheduler = getattr(
        env.curriculum_manager.cfg, difficulty_term_str
    ).func
    frac = difficulty_term.difficulty_frac
    if frac < 0.1:
        return base_mdp.modify_env_param.NO_CHANGE

    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)

    return _recurse(initial_value_tensor.tolist(), final_value_tensor.tolist(), data, frac)


def _recurse(initial_value, final_value, data, frac):
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return type(data)(
            _recurse(initial_child, final_child, data_child, frac)
            for initial_child, final_child, data_child in zip(initial_value, final_value, data, strict=False)
        )

    new_value = frac * (final_value - initial_value) + initial_value
    if isinstance(data, int):
        return int(new_value.item())
    return new_value.item()


class DeformableCommandDifficultyScheduler(ManagerTermBase):
    """Adaptive difficulty scheduler driven by the deformable command success metric."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        init_difficulty = self.cfg.params.get("init_difficulty", 0)
        self.current_adr_difficulties = torch.ones(env.num_envs, device=env.device) * init_difficulty
        self.difficulty_frac = 0.0

    def get_state(self):
        return self.current_adr_difficulties

    def set_state(self, state: torch.Tensor):
        self.current_adr_difficulties = state.clone().to(self._env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 50,
        promotion_only: bool = False,
    ):
        command = env.command_manager.get_term(command_name)
        succeeded = command.metrics["success_rate"][env_ids] > 0.5
        demoted = (
            self.current_adr_difficulties[env_ids]
            if promotion_only
            else self.current_adr_difficulties[env_ids] - 1
        )
        self.current_adr_difficulties[env_ids] = torch.where(
            succeeded,
            self.current_adr_difficulties[env_ids] + 1,
            demoted,
        ).clamp(min=min_difficulty, max=max_difficulty)
        self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)
        return self.difficulty_frac
