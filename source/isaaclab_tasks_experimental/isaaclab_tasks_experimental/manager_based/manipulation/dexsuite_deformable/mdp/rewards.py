# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for native deformable Kuka/Allegro manipulation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

from .active_deformable import active_node_data, segmented_max, segmented_min
from .soft_contacts import soft_good_contact_mask

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


def _soft_contact_gate(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    contact_body_name_groups: Sequence[str | Sequence[str]] | None,
    thumb_slot: int,
    contact_threshold: float,
) -> torch.Tensor | None:
    """Return a thumb-plus-finger soft-contact gate when configured."""
    if contact_body_name_groups is None:
        return None
    return soft_good_contact_mask(env, robot_cfg, contact_body_name_groups, thumb_slot, contact_threshold).float()


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
    nodal_pos_w, node_env_ids, _ = active_node_data(asset)
    distances = torch.linalg.norm(fingertip_pos_w[node_env_ids] - nodal_pos_w.unsqueeze(1), dim=-1)
    distance = segmented_min(distances, node_env_ids, env.num_envs)
    proximity = 1.0 - torch.tanh(distance / std)
    return proximity.mean(dim=-1)


def fingertip_deformable_reach(
    env: ManagerBasedRLEnv,
    std: float,
    fingertip_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    contact_body_name_groups: Sequence[str | Sequence[str]] | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    thumb_slot: int = 3,
    contact_threshold: float = 1.0,
    no_contact_scale: float = 1.0,
) -> torch.Tensor:
    """Reward all selected hand bodies reaching the deformable node cloud.

    Unlike the earlier proximity reward, this uses the maximum selected-body
    distance so one close finger cannot hide another finger staying far away.
    When contact groups are supplied, the term can be down-scaled until a true
    thumb-plus-finger deformable contact is present, matching the rigid
    Dexsuite reach/contact progression.
    """
    robot: Articulation = env.scene[fingertip_cfg.name]
    asset: DeformableObject = env.scene[asset_cfg.name]
    fingertip_pos_w = robot.data.body_pos_w.torch[:, fingertip_cfg.body_ids]
    nodal_pos_w, node_env_ids, _ = active_node_data(asset)
    distances = torch.linalg.norm(fingertip_pos_w[node_env_ids] - nodal_pos_w.unsqueeze(1), dim=-1)
    max_distance = segmented_min(distances, node_env_ids, env.num_envs).max(dim=-1).values
    reward = 1.0 - torch.tanh(max_distance / std)

    gate = _soft_contact_gate(env, robot_cfg, contact_body_name_groups, thumb_slot, contact_threshold)
    if gate is not None:
        reward = reward * gate.clamp(no_contact_scale, 1.0)
    return reward


def deformable_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_body_name_groups: Sequence[str | Sequence[str]] | None = None,
    thumb_slot: int = 3,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward if the deformable COM is above a minimum env-frame height."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    reward = (com_z > minimal_height).float()
    gate = _soft_contact_gate(env, robot_cfg, contact_body_name_groups, thumb_slot, contact_threshold)
    if gate is not None:
        reward = reward * gate
    return reward


def deformable_height_progress(
    env: ManagerBasedRLEnv,
    baseline_height: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_body_name_groups: Sequence[str | Sequence[str]] | None = None,
    thumb_slot: int = 3,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Dense COM-height progress from the reset height to the lift threshold."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    height_span = max(target_height - baseline_height, 1.0e-6)
    reward = ((com_z - baseline_height) / height_span).clamp(0.0, 1.0)
    gate = _soft_contact_gate(env, robot_cfg, contact_body_name_groups, thumb_slot, contact_threshold)
    if gate is not None:
        reward = reward * gate
    return reward


def deformable_com_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    contact_body_name_groups: Sequence[str | Sequence[str]] | None = None,
    thumb_slot: int = 3,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Tanh reward for deformable COM tracking the commanded position."""
    robot: Articulation = env.scene[robot_cfg.name]
    asset: DeformableObject = env.scene[asset_cfg.name]
    command_b = env.command_manager.get_command(command_name)
    command_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command_b)

    com_w = asset.data.root_pos_w.torch
    com_z = com_w[:, 2] - env.scene.env_origins[:, 2]
    distance = torch.linalg.norm(command_w - com_w, dim=1)
    reward = (com_z > minimal_height).float() * (1.0 - torch.tanh(distance / std))
    gate = _soft_contact_gate(env, robot_cfg, contact_body_name_groups, thumb_slot, contact_threshold)
    if gate is not None:
        reward = reward * gate
    return reward


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
    nodal_vel, node_env_ids, counts = active_node_data(asset, velocity=True)
    squared = torch.square(nodal_vel).mean(dim=-1)
    result = torch.zeros(env.num_envs, device=squared.device).index_add_(0, node_env_ids, squared)
    return (result / counts.clamp_min(1)).clamp(0.0, 1000.0)


def deformable_spread_l2(
    env: ManagerBasedRLEnv,
    nominal_extent: tuple[float, float, float] | None,
    margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Penalty for excessive stretching or collapse of the deformable node cloud."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos, node_env_ids, _ = active_node_data(asset)
    extent = segmented_max(nodal_pos, node_env_ids, env.num_envs) - segmented_min(nodal_pos, node_env_ids, env.num_envs)
    if nominal_extent is None:
        # Heterogeneous assets have distinct rest aspect ratios.  Their default
        # nodal state is captured after the matching clone prototype is added
        # to Newton, so it provides one rest extent per environment.
        default_pos, default_env_ids, _ = active_node_data(asset, default=True)
        target_extent = segmented_max(default_pos, default_env_ids, env.num_envs) - segmented_min(
            default_pos, default_env_ids, env.num_envs
        )
    else:
        target_extent = torch.tensor(nominal_extent, device=extent.device, dtype=extent.dtype)
    excess = torch.relu(torch.abs(extent - target_extent) - margin)
    return torch.sum(torch.square(excess), dim=1).clamp(0.0, 1000.0)
