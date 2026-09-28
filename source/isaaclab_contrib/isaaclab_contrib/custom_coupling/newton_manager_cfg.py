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


@configclass
class CoupledMJWarpSimplicitsSolverCfg(NewtonSolverCfg):
    """MJWarp rigid bodies coupled to Kaolin's experimental Simplicits solver.

    The Simplicits asset list and local poses are deliberately part of this
    solver configuration: unlike a USD TetMesh, a Kaolin reduced object is
    constructed before the Newton model is finalized.  This keeps the optional
    Kaolin dependency out of the generic Isaac Lab asset API while allowing a
    task to build one object for every environment/slot.
    """

    class_type: type[NewtonManager] | str = (
        "{DIR}.coupled_mjwarp_simplicits_manager:NewtonCoupledMJWarpSimplicitsManager"
    )
    """Manager class implementing the mixed MJWarp/Simplicits step."""

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """MJWarp solver for robots and rigid scene geometry."""

    asset_paths: list[str] = []
    """One RKPM-packaged Kaolin USD path for every Gaussian-twin slot."""

    slot_positions: list[tuple[float, float, float]] = []
    """Object poses relative to each environment origin, in slot order."""

    num_newton_steps: int = 2
    """Fixed Simplicits Newton iterations per physics substep."""

    cg_iterations: int = 32
    """Maximum conjugate-gradient iterations used by the reduced Simplicits solve."""

    cg_tolerance: float = 1.0e-6
    """Absolute residual tolerance for the reduced conjugate-gradient solve."""

    newton_convergence_tolerance: float = 1.0e-9
    """Directional-energy tolerance used to terminate each reduced Newton solve."""

    collision_particle_radius: float = 0.004
    """Newton collision radius for Simplicits quadrature particles [m]."""

    soft_contact_ke: float = 8.0e3
    """Simplicits/rigid contact stiffness [N/m]."""

    soft_contact_mu: float = 1.0
    """Simplicits/rigid contact friction coefficient."""

    soft_contact_coefficient: float = 0.05
    """Weight applied to Kaolin's Newton-shape soft-contact energy."""

    enable_inter_object_collisions: bool = False
    """Enable Kaolin RKPM soft/soft contact between concurrently active twin slots."""

    max_quadrature_points: int | None = 2048
    """Optional deterministic quadrature-point cap per packaged object."""
