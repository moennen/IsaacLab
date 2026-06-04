# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generators for deformable-object goal positions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


POSITION_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "goal_far": sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.10), opacity=0.45),
        ),
        "goal_near": sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.85, 0.20), opacity=0.45),
        ),
        "current_far": sim_utils.SphereCfg(
            radius=0.018,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.95), opacity=0.70),
        ),
        "current_near": sim_utils.SphereCfg(
            radius=0.018,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.85, 0.20), opacity=0.70),
        ),
    }
)


class DeformableUniformPositionCommand(CommandTerm):
    """Uniform goal position sampled in the robot root frame for a deformable COM."""

    cfg: DeformableUniformPositionCommandCfg

    def __init__(self, cfg: DeformableUniformPositionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.deformable: DeformableObject = env.scene[cfg.deformable_name]

        self.position_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.position_command_w = torch.zeros_like(self.position_command_b)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)

        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/position"
        self.cfg.element_names = self.cfg.element_names or ["x", "y", "z"]

    @property
    def command(self) -> torch.Tensor:
        """Desired deformable COM position in the robot root frame."""
        return self.position_command_b

    def __str__(self) -> str:
        msg = "DeformableUniformPositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    def _update_metrics(self) -> None:
        self.position_command_w, _ = combine_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.position_command_b,
        )
        com_w = self.deformable.data.root_pos_w.torch
        self.metrics["position_error"] = torch.linalg.norm(self.position_command_w - com_w, dim=-1)
        self.metrics["success_rate"] = (self.metrics["position_error"] < self.cfg.success_threshold).float()

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        r = torch.empty(len(env_ids), device=self.device)
        self.position_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.position_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.position_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

    def _update_command(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "position_visualizer"):
                from isaaclab.markers import VisualizationMarkers

                self.position_visualizer = VisualizationMarkers(self.cfg.position_visualizer_cfg)
            self.position_visualizer.set_visibility(True)
        elif hasattr(self, "position_visualizer"):
            self.position_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized:
            return

        success = self.metrics["position_error"] < self.cfg.success_threshold
        translations = torch.cat((self.position_command_w, self.deformable.data.root_pos_w.torch), dim=0)
        marker_indices = torch.cat((success.long(), success.long() + 2), dim=0)
        self.position_visualizer.visualize(translations=translations, marker_indices=marker_indices)


@configclass
class DeformableUniformPositionCommandCfg(CommandTermCfg):
    """Configuration for :class:`DeformableUniformPositionCommand`."""

    class_type: type[DeformableUniformPositionCommand] = DeformableUniformPositionCommand

    asset_name: str = MISSING
    """Name of the robot asset that defines the command frame."""

    deformable_name: str = MISSING
    """Name of the deformable object whose COM should track the command."""

    success_threshold: float = 0.06
    """Distance threshold [m] used for command success metrics and debug coloring."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for sampled target positions."""

        pos_x: tuple[float, float] = MISSING
        """Range for x position [m] in the robot root frame."""

        pos_y: tuple[float, float] = MISSING
        """Range for y position [m] in the robot root frame."""

        pos_z: tuple[float, float] = MISSING
        """Range for z position [m] in the robot root frame."""

    ranges: Ranges = MISSING
    """Ranges for the sampled goal positions."""

    position_visualizer_cfg: VisualizationMarkersCfg = POSITION_MARKER_CFG.replace(
        prim_path="/Visuals/Command/deformable_position"
    )
    """Visualization marker used for goal and current deformable COM positions."""
