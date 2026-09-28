# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-world host MuJoCo through Newton, with standard robot state adapters."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonManager
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager

from isaaclab.utils.configclass import configclass


@configclass
class TomatoSolverCfg(MJWarpSolverCfg):
    """Reduce compliant tangential creep in the small-fruit rubber-pad grasp."""

    impratio: float = 10.0


class NewtonTomatoManager(NewtonMJWarpManager):
    """Use Newton's native double-precision solver for the slender plant.

    Newton's CPU MuJoCo backend resolves contacts internally but does not export
    its contact array. Frame/IK state and weld reactions remain available. The
    task therefore does not create Newton contact sensors on this backend.
    """

    @classmethod
    def _initialize_contacts(cls):
        if cls._solver.use_mujoco_cpu:
            if cls._report_contacts:
                raise ValueError("CPU tomato task does not support Newton contact sensor export.")
            NewtonManager._contacts = None
        else:
            super()._initialize_contacts()

    @classmethod
    def _reset_solver_internals(cls, world_mask):
        super()._reset_solver_internals(world_mask)
        if world_mask is not None and world_mask.numpy().any():
            solver = cls._solver
            data = solver.mj_data if solver.use_mujoco_cpu else solver.mjw_data
            solver._update_mjc_data(data, cls._model, cls._state_0)
