# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for native deformable Kuka/Allegro manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action rate while bounding pathological spikes."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(0.0, 1000.0)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize absolute action magnitude while bounding pathological spikes."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(0.0, 1000.0)


def fingertip_deformable_proximity(
    env: ManagerBasedRLEnv,
    std: float,
    fingertip_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Reward fingertips being near deformable material points."""
    robot: Articulation = env.scene[fingertip_cfg.name]
    asset: DeformableObject = env.scene[asset_cfg.name]
    fingertip_pos_w = robot.data.body_pos_w.torch[:, fingertip_cfg.body_ids]
    nodal_pos_w = asset.data.nodal_pos_w.torch
    distance = torch.linalg.norm(fingertip_pos_w.unsqueeze(2) - nodal_pos_w.unsqueeze(1), dim=-1).min(dim=-1).values
    proximity = 1.0 - torch.tanh(distance / std)
    return proximity.mean(dim=-1)


def deformable_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Reward if the deformable COM is above a minimum env-frame height."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (com_z > minimal_height).float()


def deformable_height_progress(
    env: ManagerBasedRLEnv,
    baseline_height: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Dense COM-height progress from the reset height to the lift threshold."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    height_span = max(target_height - baseline_height, 1.0e-6)
    return ((com_z - baseline_height) / height_span).clamp(0.0, 1.0)


def deformable_com_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Tanh reward for deformable COM tracking the commanded position."""
    robot: Articulation = env.scene[robot_cfg.name]
    asset: DeformableObject = env.scene[asset_cfg.name]
    command_b = env.command_manager.get_command(command_name)
    command_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command_b)

    com_w = asset.data.root_pos_w.torch
    com_z = com_w[:, 2] - env.scene.env_origins[:, 2]
    distance = torch.linalg.norm(command_w - com_w, dim=1)
    return (com_z > minimal_height).float() * (1.0 - torch.tanh(distance / std))


def fingertip_below_height(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    fingertip_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize fingertips scraping below a table/object clearance height."""
    robot: Articulation = env.scene[fingertip_cfg.name]
    fingertip_z = robot.data.body_pos_w.torch[:, fingertip_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    return torch.relu(minimum_height - fingertip_z).mean(dim=1)


def deformable_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Mean squared nodal velocity penalty."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    return torch.mean(torch.square(asset.data.nodal_vel_w.torch), dim=(1, 2)).clamp(0.0, 1000.0)


def deformable_spread_l2(
    env: ManagerBasedRLEnv,
    nominal_extent: tuple[float, float, float],
    margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Penalty for excessive stretching or collapse of the deformable node cloud."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = asset.data.nodal_pos_w.torch
    extent = nodal_pos.max(dim=1).values - nodal_pos.min(dim=1).values
    target_extent = torch.tensor(nominal_extent, device=extent.device, dtype=extent.dtype)
    excess = torch.relu(torch.abs(extent - target_extent) - margin)
    return torch.sum(torch.square(excess), dim=1).clamp(0.0, 1000.0)
