# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for native deformable Kuka/Allegro manipulation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


def _finite(tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    """Return a finite tensor without mutating the physics data view."""
    return torch.nan_to_num(tensor, nan=value, posinf=value, neginf=value)


def deformable_com_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Deformable COM position in the robot root frame."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    com_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        asset.data.root_pos_w.torch,
    )
    return _finite(com_b)


def deformable_root_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Deformable root velocity in the robot root frame."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    vel_b = quat_apply_inverse(robot.data.root_quat_w.torch, asset.data.root_vel_w.torch)
    return _finite(vel_b)


def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body pose and twist in the base asset root frame.

    Per selected body, the output is ``[pos(3), quat(4), lin_vel(3), ang_vel(3)]``.
    """
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]

    body_pos_w = body_asset.data.body_pos_w.torch[:, body_asset_cfg.body_ids].reshape(-1, 3)
    body_quat_w = body_asset.data.body_quat_w.torch[:, body_asset_cfg.body_ids].reshape(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w.torch[:, body_asset_cfg.body_ids].reshape(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w.torch[:, body_asset_cfg.body_ids].reshape(-1, 3)

    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    root_pos_w = base_asset.data.root_link_pos_w.torch.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.torch.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)

    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)

    state = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=-1)
    return _finite(state.reshape(env.num_envs, -1))


def fingertip_deformable_distances(
    env: ManagerBasedRLEnv,
    fingertip_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Nearest nodal distance for each selected fingertip."""
    robot: Articulation = env.scene[fingertip_cfg.name]
    asset: DeformableObject = env.scene[asset_cfg.name]
    fingertip_pos_w = robot.data.body_pos_w.torch[:, fingertip_cfg.body_ids]
    nodal_pos_w = asset.data.nodal_pos_w.torch
    distances = torch.linalg.norm(fingertip_pos_w.unsqueeze(2) - nodal_pos_w.unsqueeze(1), dim=-1)
    return _finite(distances.min(dim=-1).values)


def deformable_extent_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Axis-aligned nodal extent in the robot root frame."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    nodal_pos_w = asset.data.nodal_pos_w.torch
    num_nodes = nodal_pos_w.shape[1]
    root_pos_w = robot.data.root_pos_w.torch.unsqueeze(1).expand(-1, num_nodes, -1).reshape(-1, 3)
    root_quat_w = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, num_nodes, -1).reshape(-1, 4)
    nodal_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, nodal_pos_w.reshape(-1, 3))
    nodal_pos_b = nodal_pos_b.reshape(env.num_envs, num_nodes, 3)
    extent = nodal_pos_b.max(dim=1).values - nodal_pos_b.min(dim=1).values
    return _finite(extent)


def time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Remaining normalized episode time."""
    return (env.max_episode_length - env.episode_length_buf).unsqueeze(-1) / env.max_episode_length


class DeformableSampledNodesInRobotRootFrame(ManagerTermBase):
    """Fixed material-node samples expressed in the robot root frame.

    Node IDs are sampled at reset and kept fixed for the episode, so the policy
    observes motion of consistent material points instead of a resampled cloud.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("deformable"))
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.num_nodes: int = cfg.params.get("num_nodes", 32)
        self.include_velocities: bool = cfg.params.get("include_velocities", True)

        asset: DeformableObject = env.scene[self.asset_cfg.name]
        self.total_nodes = asset.data.nodal_pos_w.shape[1]
        self.node_ids = torch.empty(env.num_envs, self.num_nodes, dtype=torch.long, device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resample observed material nodes for the selected environments."""
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        if self.num_nodes <= self.total_nodes:
            self.node_ids[env_ids] = (
                torch.rand((num_envs, self.total_nodes), device=self.device).topk(self.num_nodes, dim=1).indices
            )
        else:
            self.node_ids[env_ids] = torch.randint(self.total_nodes, (num_envs, self.num_nodes), device=self.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        num_nodes: int = 32,
        include_velocities: bool = True,
    ) -> torch.Tensor:
        """Return flattened sampled node positions and optionally velocities."""
        if num_nodes != self.num_nodes:
            raise ValueError(f"Requested {num_nodes} nodes, but term was initialized with {self.num_nodes}.")
        if include_velocities != self.include_velocities:
            raise ValueError(
                f"Requested include_velocities={include_velocities}, but term was initialized with"
                f" {self.include_velocities}."
            )

        asset: DeformableObject = env.scene[asset_cfg.name]
        robot: Articulation = env.scene[robot_cfg.name]

        gather_ids = self.node_ids.unsqueeze(-1).expand(-1, -1, 3)
        sampled_pos_w = asset.data.nodal_pos_w.torch.gather(1, gather_ids)

        root_pos_w = robot.data.root_pos_w.torch.unsqueeze(1).expand(-1, self.num_nodes, -1).reshape(-1, 3)
        root_quat_w = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, self.num_nodes, -1).reshape(-1, 4)
        sampled_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, sampled_pos_w.reshape(-1, 3))
        sampled_pos_b = sampled_pos_b.reshape(env.num_envs, self.num_nodes, 3)

        if not self.include_velocities:
            return _finite(sampled_pos_b.reshape(env.num_envs, -1))

        sampled_vel_w = asset.data.nodal_vel_w.torch.gather(1, gather_ids)
        sampled_vel_b = quat_apply_inverse(root_quat_w, sampled_vel_w.reshape(-1, 3))
        sampled_vel_b = sampled_vel_b.reshape(env.num_envs, self.num_nodes, 3)
        sampled_state_b = torch.cat((sampled_pos_b, sampled_vel_b), dim=-1)
        return _finite(sampled_state_b.reshape(env.num_envs, -1))
