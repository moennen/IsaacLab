# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for native deformable Kuka/Allegro manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


def deformable_com_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Terminate when the deformable COM falls below a minimum env-frame height."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return com_z < minimum_height


def deformable_nodal_out_of_bounds(
    env: ManagerBasedRLEnv,
    in_bound_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Terminate when any deformable node leaves the configured env-frame bounds."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = asset.data.nodal_pos_w.torch - env.scene.env_origins.unsqueeze(1)
    ranges = torch.tensor(
        [in_bound_range.get(axis, (-float("inf"), float("inf"))) for axis in ("x", "y", "z")],
        device=nodal_pos.device,
        dtype=nodal_pos.dtype,
    )
    outside = (nodal_pos < ranges[:, 0]) | (nodal_pos > ranges[:, 1])
    return outside.any(dim=(1, 2))


def deformable_state_invalid(
    env: ManagerBasedRLEnv,
    max_velocity: float,
    max_extent: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Terminate on non-finite or clearly unstable deformable state."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = asset.data.nodal_pos_w.torch
    nodal_vel = asset.data.nodal_vel_w.torch
    nonfinite = (~torch.isfinite(nodal_pos)).any(dim=(1, 2)) | (~torch.isfinite(nodal_vel)).any(dim=(1, 2))
    excessive_velocity = torch.linalg.norm(nodal_vel, dim=-1).max(dim=1).values > max_velocity
    extent = nodal_pos.max(dim=1).values - nodal_pos.min(dim=1).values
    excessive_extent = extent.max(dim=1).values > max_extent
    return nonfinite | excessive_velocity | excessive_extent


def abnormal_robot_state(
    env: ManagerBasedRLEnv,
    velocity_limit_scale: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate on robot joint velocities beyond scaled soft limits."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel.torch
    joint_vel_limits = robot.data.joint_vel_limits.torch
    return (joint_vel.abs() > (joint_vel_limits * velocity_limit_scale)).any(dim=1)
