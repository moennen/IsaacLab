# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Relative IK with a TCP offset expressed in the Jacobian's base frame."""

import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import math as math_utils


class TomatoIKAction(DifferentialInverseKinematicsAction):
    """Rotate the hand-to-TCP displacement before shifting its spatial Jacobian."""

    def _compute_frame_jacobian(self):
        self._jacobian_b[:] = self.jacobian_b
        if self.cfg.body_offset is not None:
            body_quat = self._asset.data.body_quat_w.torch[:, self._body_idx]
            base_quat = self._asset.data.root_quat_w.torch
            hand_in_base = math_utils.quat_mul(math_utils.quat_inv(base_quat), body_quat)
            offset_in_base = math_utils.quat_apply(hand_in_base, self._offset_pos)
            self._jacobian_b[:, :3, :] -= torch.bmm(
                math_utils.skew_symmetric_matrix(offset_in_base), self._jacobian_b[:, 3:, :]
            )
        return self._jacobian_b
