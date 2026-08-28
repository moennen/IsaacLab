# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
from pathlib import Path

from isaaclab_newton.physics import HydroelasticSDFCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import MeshCollisionBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack.stack_env_cfg import PhysicsCfg, mdp

from . import stack_joint_pos_env_cfg

_DEFAULT_ALIGNED_BACKGROUND_USD = Path(__file__).resolve().parents[7] / "nova_carter-galileo.usda"
_LOGGER = logging.getLogger(__name__)


def _spawn_aligned_background(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Add the alignment layer as a root sublayer, preserving its authored world transforms."""
    from isaaclab.sim import get_current_stage

    stage = get_current_stage()
    root_layer = stage.GetRootLayer()
    background_path = str(Path(cfg.usd_path).resolve())
    if background_path not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(background_path)
    # The alignment layer already contains the authored transform. Keep the spawn API contract
    # explicit and apply non-identity initial-state transforms when a caller provides them.
    if translation not in (None, (0.0, 0.0, 0.0)) or orientation not in (None, (0.0, 0.0, 0.0, 1.0)):
        from pxr import Gf, UsdGeom

        background = stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(background)
        ops = xform.GetOrderedXformOps()
        translate_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        orient_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
        if translation != (0.0, 0.0, 0.0):
            if translate_op is None:
                translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(*translation))
        if orientation != (0.0, 0.0, 0.0, 1.0):
            if orient_op is None:
                orient_op = xform.AddOrientOp()
            orient_op.Set(Gf.Quatd(orientation[3], Gf.Vec3d(*orientation[:3])))
    return stage.GetPrimAtPath(prim_path)


##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import (  # isort: skip
    FRANKA_PANDA_HIGH_PD_CFG,
)


@configclass
class FrankaCubeStackEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # Use a stiffer PD controller for better IK tracking.
        robot_init_state = self.scene.robot.init_state
        robot_semantic_tags = self.scene.robot.spawn.semantic_tags
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot", init_state=robot_init_state
        )
        self.scene.robot.spawn.semantic_tags = robot_semantic_tags

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )


@configclass
class FrankaCubeStackNewtonEnvCfg(FrankaCubeStackEnvCfg):
    """Newton-specific Franka stack configuration for interactive teleoperation."""

    def __post_init__(self):
        super().__post_init__()

        aligned_background = Path(
            os.environ.get("ISAACLAB_ALIGNED_BACKGROUND_USD", str(_DEFAULT_ALIGNED_BACKGROUND_USD))
        ).expanduser()
        if aligned_background.is_file():
            self.scene.background = AssetBaseCfg(
                prim_path="/World/GaussianBackground",
                spawn=UsdFileCfg(
                    func=_spawn_aligned_background,
                    usd_path=str(aligned_background),
                ),
            )
        else:
            _LOGGER.warning("Aligned background USDA not found at '%s'; continuing without it.", aligned_background)

        # Newton currently does not honor PhysX per-body gravity disable.
        # MuJoCo-style gravity compensation keeps the Franka fixed under gravity.
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation

        newton_physics = PhysicsCfg().newton_mjwarp
        newton_physics.num_substeps = 4
        # The hydroelastic gripper contact manifold can exceed the default MuJoCo
        # constraint budget during cube contact.
        newton_physics.solver_cfg.njmax = 600
        newton_physics.solver_cfg.nconmax = 400
        newton_physics.collision_cfg = NewtonCollisionPipelineCfg(
            soft_contact_max=0,
            sdf_hydroelastic_config=HydroelasticSDFCfg(
                reduce_contacts=True,
                normal_matching=True,
                anchor_contact=True,
            ),
        )
        newton_physics.default_shape_cfg.gap = 0.005
        newton_physics.default_shape_cfg.ke = 1.0e6
        newton_physics.default_shape_cfg.kd = 2.0e3
        self.sim.physics = newton_physics

        contact_props = sim_utils.NewtonSDFCollisionPropertiesCfg(
            contact_gap=0.005,
            rest_offset=0.0,
            sdf_max_resolution=64,
            sdf_narrow_band_inner=-0.005,
            sdf_narrow_band_outer=0.005,
            hydroelastic_enabled=True,
            hydroelastic_stiffness=1.0e11,
            # Preserve SDF meshes: the Franka USD authors convexHull by default,
            # which conflicts with Newton hydroelastic SDF generation.
            mesh_collision_property=MeshCollisionBaseCfg(mesh_approximation_name="none"),
        )
        contact_material = [
            UsdPhysicsRigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0),
            NewtonMaterialCfg(contact_stiffness=1.0e6, contact_damping=2.0e3),
        ]
        # Newton must edit the individual finger collision meshes to attach SDF schemas.
        # The stock Franka asset is instanceable, so opt out for this Newton variant.
        self.scene.robot.spawn.make_uninstanceable = True
        self.scene.robot.spawn.collision_props = contact_props
        self.scene.robot.spawn.physics_material = contact_material
        self.scene.robot.spawn.rigid_props = sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0)
        # Newton hydroelastic contacts need a compliant, force-limited gripper.
        # The inherited high-PD hand (2e3 Nm/rad, 200 N) launches cubes on contact.
        self.scene.robot.actuators["panda_hand"].stiffness = 80.0
        self.scene.robot.actuators["panda_hand"].damping = 40.0
        self.scene.robot.actuators["panda_hand"].joint_effort_limit = 35.0
        cube_contact_props = contact_props.replace(
            mesh_collision_property=MeshCollisionBaseCfg(mesh_approximation_name="sdf")
        )
        for cube in (self.scene.cube_1, self.scene.cube_2, self.scene.cube_3):
            cube.spawn.collision_props = cube_contact_props
            cube.spawn.physics_material = contact_material


@configclass
class FrankaCubeStackRedGreenEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={
                "cube_1_cfg": SceneEntityCfg("cube_2"),
                "cube_2_cfg": SceneEntityCfg("cube_3"),
                "cube_3_cfg": None,
            },
        )


@configclass
class FrankaCubeStackRedGreenBlueEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={
                "cube_1_cfg": SceneEntityCfg("cube_2"),
                "cube_2_cfg": SceneEntityCfg("cube_3"),
                "cube_3_cfg": SceneEntityCfg("cube_1"),
            },
        )


@configclass
class FrankaCubeStackBlueGreenEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={
                "cube_1_cfg": SceneEntityCfg("cube_1"),
                "cube_2_cfg": SceneEntityCfg("cube_3"),
                "cube_3_cfg": None,
            },
        )


@configclass
class FrankaCubeStackBlueGreenRedEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={
                "cube_1_cfg": SceneEntityCfg("cube_1"),
                "cube_2_cfg": SceneEntityCfg("cube_3"),
                "cube_3_cfg": SceneEntityCfg("cube_2"),
            },
        )
