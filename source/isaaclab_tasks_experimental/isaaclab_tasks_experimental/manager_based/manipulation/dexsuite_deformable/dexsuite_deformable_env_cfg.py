# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental Kuka/Allegro task using native Newton deformable coupling.

This task intentionally restarts the old deformable Dexsuite prototype around
native :class:`DeformableObjectCfg` state.  The trainable surface is COM lifting
and position control; no synthetic deformable orientation is exposed.
"""

from __future__ import annotations

from isaaclab_newton.physics import FeatherstoneSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledFeatherstoneVBDSolverCfg,
    CoupledMJWarpVBDSolverCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from . import mdp
from .spawners import NewtonTetCuboidCfg

FINGERTIP_LIST = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
DEFORMABLE_SIZE = (0.09, 0.08, 0.07)
DEFORMABLE_INIT_POS = (-0.55, 0.10, 0.34)
TABLE_POS = (-0.55, 0.0, 0.235)
TABLE_TOP_Z = TABLE_POS[2] + 0.02
YOUNGS_MODULUS = 6.0e4
POISSONS_RATIO = 0.25
SOFT_CONTACT_MAX = 1_048_576


TABLE_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(0.8, 1.5, 0.04),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.38, 0.40, 0.42)),
)


def _rigid_solver(njmax: int, nconmax: int, ls_iterations: int, impratio: float) -> MJWarpSolverCfg:
    """Create a conservative rigid solver config for coupled soft contact."""
    return MJWarpSolverCfg(
        njmax=njmax,
        nconmax=nconmax,
        ls_iterations=ls_iterations,
        cone="pyramidal",
        impratio=impratio,
        ls_parallel=False,
        integrator="implicitfast",
        ccd_iterations=100,
    )


def _soft_solver(iterations: int) -> VBDSolverCfg:
    """Create the VBD soft-body solver config."""
    return VBDSolverCfg(
        iterations=iterations,
        integrate_with_external_rigid_solver=True,
        particle_enable_self_contact=False,
        particle_collision_detection_interval=-1,
    )


def _coupled_newton_cfg(
    *,
    coupling_mode: str,
    num_substeps: int,
    rigid_solver_cfg: MJWarpSolverCfg,
    soft_solver_cfg: VBDSolverCfg,
    model_cfg: NewtonModelCfg,
) -> NewtonCfg:
    """Create a kitless Newton config with coupled deformable model parameters."""
    cfg = NewtonCfg(
        solver_cfg=CoupledMJWarpVBDSolverCfg(
            rigid_solver_cfg=rigid_solver_cfg,
            soft_solver_cfg=soft_solver_cfg,
            coupling_mode=coupling_mode,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=SOFT_CONTACT_MAX),
        num_substeps=num_substeps,
        use_cuda_graph=True,
    )
    # The coupled manager checks this optional attribute after finalizing the model.
    cfg.model_cfg = model_cfg
    return cfg


def _kinematic_newton_cfg(
    *,
    num_substeps: int,
    soft_solver_cfg: VBDSolverCfg,
    model_cfg: NewtonModelCfg,
    velocity_limit_scale: float,
) -> NewtonCfg:
    """Create the Newton example-style Featherstone kinematic + VBD config."""
    cfg = NewtonCfg(
        solver_cfg=CoupledFeatherstoneVBDSolverCfg(
            rigid_solver_cfg=FeatherstoneSolverCfg(update_mass_matrix_interval=num_substeps),
            soft_solver_cfg=soft_solver_cfg,
            coupling_mode="kinematic",
            kinematic_velocity_limit_scale=velocity_limit_scale,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=SOFT_CONTACT_MAX),
        num_substeps=num_substeps,
        use_cuda_graph=True,
    )
    cfg.model_cfg = model_cfg
    return cfg


DEFORMABLE_MODEL_CFG = NewtonModelCfg(
    soft_contact_ke=8.0e3,
    soft_contact_kd=1.0e-5,
    soft_contact_mu=4.0,
    shape_material_ke=3.0e4,
    shape_material_kd=1.0e-5,
    shape_material_mu=4.0,
)


@configclass
class PhysicsCfg(PresetCfg):
    """Physics presets for stability/performance sweeps."""

    stable_kinematic: NewtonCfg = _kinematic_newton_cfg(
        num_substeps=8,
        soft_solver_cfg=_soft_solver(iterations=12),
        model_cfg=DEFORMABLE_MODEL_CFG,
        velocity_limit_scale=0.75,
    )

    fast_kinematic: NewtonCfg = _kinematic_newton_cfg(
        num_substeps=4,
        soft_solver_cfg=_soft_solver(iterations=7),
        model_cfg=DEFORMABLE_MODEL_CFG,
        velocity_limit_scale=1.0,
    )

    stable_two_way: NewtonCfg = _coupled_newton_cfg(
        coupling_mode="two_way",
        num_substeps=8,
        rigid_solver_cfg=_rigid_solver(njmax=512, nconmax=128, ls_iterations=24, impratio=5.0),
        soft_solver_cfg=_soft_solver(iterations=12),
        model_cfg=DEFORMABLE_MODEL_CFG,
    )

    fast_two_way: NewtonCfg = _coupled_newton_cfg(
        coupling_mode="two_way",
        num_substeps=4,
        rigid_solver_cfg=_rigid_solver(njmax=400, nconmax=96, ls_iterations=14, impratio=4.0),
        soft_solver_cfg=_soft_solver(iterations=7),
        model_cfg=DEFORMABLE_MODEL_CFG,
    )

    one_way_debug: NewtonCfg = _coupled_newton_cfg(
        coupling_mode="one_way",
        num_substeps=8,
        rigid_solver_cfg=_rigid_solver(njmax=512, nconmax=128, ls_iterations=24, impratio=5.0),
        soft_solver_cfg=_soft_solver(iterations=12),
        model_cfg=DEFORMABLE_MODEL_CFG,
    )

    default = stable_kinematic


DEFORMABLE_OBJECT_CFG = DeformableObjectCfg(
    prim_path="/World/envs/env_.*/Deformable",
    init_state=DeformableObjectCfg.InitialStateCfg(pos=DEFORMABLE_INIT_POS),
    spawn=NewtonTetCuboidCfg(
        size=DEFORMABLE_SIZE,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.16, 0.22)),
        physics_material=NewtonDeformableBodyMaterialCfg(
            density=300.0,
            k_mu=YOUNGS_MODULUS / (2.0 * (1.0 + POISSONS_RATIO)),
            k_lambda=YOUNGS_MODULUS
            * POISSONS_RATIO
            / ((1.0 + POISSONS_RATIO) * (1.0 - 2.0 * POISSONS_RATIO)),
            particle_radius=0.012,
        ),
    ),
)


@configclass
class KukaAllegroDeformableSceneCfg(InteractiveSceneCfg):
    """Kuka/Allegro scene with a native Newton deformable object."""

    robot: ArticulationCfg = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    deformable: DeformableObjectCfg = DEFORMABLE_OBJECT_CFG

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=TABLE_SPAWN_CFG,
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_POS, rot=(0.0, 0.0, 0.0, 1.0)),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=GroundPlaneCfg(color=(0.95, 0.95, 0.95)),
        collision_group=-1,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    def __post_init__(self) -> None:
        # Reduce aggressive depenetration spikes when the fingers first couple to the soft body.
        self.robot.spawn.rigid_props.max_depenetration_velocity = 5.0


@configclass
class CommandsCfg:
    """Command terms for the deformable COM goal."""

    deformable_position = mdp.DeformableUniformPositionCommandCfg(
        asset_name="robot",
        deformable_name="deformable",
        resampling_time_range=(2.0, 3.0),
        debug_vis=False,
        success_threshold=0.07,
        ranges=mdp.DeformableUniformPositionCommandCfg.Ranges(
            pos_x=(-0.70, -0.35),
            pos_y=(-0.22, 0.30),
            pos_z=(0.50, 0.75),
        ),
    )


@configclass
class ActionsCfg:
    """Relative joint position control with conservative soft-contact steps."""

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            "iiwa7_joint_.*": 0.035,
            "(index|middle|ring|thumb)_joint_.*": 0.050,
        },
    )


@configclass
class ObservationsCfg:
    """Observation groups for state-only deformable manipulation."""

    @configclass
    class PolicyCfg(ObsGroup):
        deformable_com = ObsTerm(func=mdp.deformable_com_b, clip=(-2.0, 2.0))
        deformable_root_vel = ObsTerm(func=mdp.deformable_root_vel_b, clip=(-10.0, 10.0))
        fingertip_distances = ObsTerm(
            func=mdp.fingertip_deformable_distances,
            clip=(0.0, 1.5),
            params={"fingertip_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_LIST)},
        )
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "deformable_position"})
        actions = ObsTerm(func=mdp.last_action)
        time_left = ObsTerm(func=mdp.time_left)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-3.2, 3.2))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-50.0, 50.0))
        fingertip_state = ObsTerm(
            func=mdp.body_state_b,
            clip=(-5.0, 5.0),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_LIST),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class DeformableCfg(ObsGroup):
        sampled_nodes = ObsTerm(
            func=mdp.DeformableSampledNodesInRobotRootFrame,
            clip=(-5.0, 5.0),
            params={"asset_cfg": SceneEntityCfg("deformable"), "num_nodes": 32, "include_velocities": True},
        )
        extent = ObsTerm(func=mdp.deformable_extent_b, clip=(0.0, 1.0))

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()
    deformable: DeformableCfg = DeformableCfg()


@configclass
class EventCfg:
    """Reset events with small perturbations around a stable pre-grasp setup."""

    reset_robot_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_robot_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names="iiwa7_joint_.*"),
        },
    )

    reset_robot_hand_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.10, 0.10),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names="(index|middle|ring|thumb)_joint_.*"),
        },
    )

    reset_deformable = EventTerm(
        func=mdp.reset_nodal_state_uniform,
        mode="reset",
        params={
            "position_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )


@configclass
class RewardsCfg:
    """Dense shaping for grasping, lifting, and goal tracking."""

    fingertip_proximity = RewTerm(
        func=mdp.fingertip_deformable_proximity,
        params={"std": 0.08, "fingertip_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_LIST)},
        weight=0.75,
    )
    height_progress = RewTerm(
        func=mdp.deformable_height_progress,
        params={
            "baseline_height": DEFORMABLE_INIT_POS[2],
            "target_height": 0.42,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=4.0,
    )
    lifting = RewTerm(
        func=mdp.deformable_lifted,
        params={"minimal_height": 0.42, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )
    goal_tracking = RewTerm(
        func=mdp.deformable_com_goal_distance,
        params={
            "std": 0.25,
            "minimal_height": 0.40,
            "command_name": "deformable_position",
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=12.0,
    )
    goal_tracking_fine = RewTerm(
        func=mdp.deformable_com_goal_distance,
        params={
            "std": 0.06,
            "minimal_height": 0.40,
            "command_name": "deformable_position",
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=4.0,
    )
    deformable_velocity = RewTerm(func=mdp.deformable_velocity_l2, weight=-0.01)
    deformable_spread = RewTerm(
        func=mdp.deformable_spread_l2,
        params={"nominal_extent": DEFORMABLE_SIZE, "margin": 0.06},
        weight=-1.0,
    )
    fingertip_table_scrape = RewTerm(
        func=mdp.fingertip_below_height,
        params={
            "minimum_height": TABLE_TOP_Z + 0.045,
            "fingertip_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_LIST),
        },
        weight=-8.0,
    )
    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.003)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.008)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    early_termination = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": "deformable_state_invalid|abnormal_robot"},
        weight=-2.0,
    )


@configclass
class TerminationsCfg:
    """Episode termination terms with explicit deformable validity checks."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    deformable_dropped = DoneTerm(
        func=mdp.deformable_com_below_minimum,
        params={"minimum_height": 0.18, "asset_cfg": SceneEntityCfg("deformable")},
    )

    deformable_out_of_bounds = DoneTerm(
        func=mdp.deformable_nodal_out_of_bounds,
        params={
            "in_bound_range": {"x": (-1.05, -0.05), "y": (-0.70, 0.70), "z": (0.05, 1.25)},
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )

    deformable_state_invalid = DoneTerm(
        func=mdp.deformable_state_invalid,
        params={"max_velocity": 25.0, "max_extent": 0.55, "asset_cfg": SceneEntityCfg("deformable")},
    )

    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state, params={"velocity_limit_scale": 2.5})


@configclass
class DexsuiteDeformableKukaAllegroLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL config for Kuka/Allegro deformable lifting."""

    viewer: ViewerCfg = ViewerCfg(eye=(-2.20, 0.10, 0.90), lookat=(-0.55, 0.05, 0.45), origin_type="env")
    scene: KukaAllegroDeformableSceneCfg = KukaAllegroDeformableSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 6.0
        self.is_finite_horizon = False

        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.sim.physics = PhysicsCfg()


@configclass
class DexsuiteDeformableKukaAllegroLiftEnvCfg_PLAY(DexsuiteDeformableKukaAllegroLiftEnvCfg):
    """Small interactive variant with command visualization enabled."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.commands.deformable_position.debug_vis = True
