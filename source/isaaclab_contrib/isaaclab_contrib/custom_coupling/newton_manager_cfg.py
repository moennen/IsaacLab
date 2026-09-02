# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the custom MJWarp and VBD coupling manager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab_newton.physics import FeatherstoneSolverCfg, MJWarpSolverCfg, NewtonSolverCfg, VBDSolverCfg

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class CoupledMJWarpVBDSolverCfg(NewtonSolverCfg):
    """Configuration for the custom MJWarp and VBD coupling manager."""

    class_type: type[NewtonManager] | str = "{DIR}.coupled_mjwarp_vbd_manager:NewtonCoupledMJWarpVBDManager"
    """Manager class for the coupled solver."""

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """MJWarp rigid-body solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD deformable solver configuration."""

    coupling_mode: Literal["one_way", "two_way"] = "two_way"
    """Coupling direction between the rigid and deformable solvers."""


@configclass
class CoupledFeatherstoneVBDSolverCfg(NewtonSolverCfg):
    """Kinematic Featherstone and externally integrated VBD coupling.

    This retained Isaac Lab coupling topology is useful as a stable policy-training
    baseline: the articulated robot is advanced from its position targets as a
    bounded kinematic collider, while VBD receives the resulting body motion but
    does not feed soft-body reactions back into the robot.
    """

    class_type: type[NewtonManager] | str = "{DIR}.coupled_featherstone_vbd_manager:NewtonCoupledFeatherstoneVBDManager"
    """Manager class implementing the kinematic coupled step."""

    rigid_solver_cfg: FeatherstoneSolverCfg = FeatherstoneSolverCfg()
    """Featherstone rigid-body solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD deformable solver configuration."""

    kinematic_velocity_limit_scale: float = 1.0
    """Multiplier applied to joint velocity limits while pursuing position targets."""

    shape_material_ke: float | None = None
    """Optional global rigid-shape contact stiffness override [N/m]."""

    shape_material_kd: float | None = None
    """Optional global rigid-shape contact damping override [N*s/m]."""

    shape_material_mu: float | None = None
    """Optional global rigid-shape friction override."""
