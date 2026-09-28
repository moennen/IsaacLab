# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-workcell relative-IK tomato picking with Newton MuJoCo contacts."""

import os
from pathlib import Path

from isaaclab_newton.physics import NewtonCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonRTXVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.gaussian_tasks.task_assets import task_asset_root
from isaaclab_tasks.contrib.stack.config.franka.stack_joint_pos_env_cfg import FrankaCubeStackEnvCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from .actions import TomatoIKAction
from .gamepad import TomatoGamepadCfg
from .keyboard import TomatoKeyboard
from .physics import TomatoSolverCfg
from .plant import TomatoPlantCfg

DEFAULT_ASSET_ROOT = str(task_asset_root("tomato"))


def fruit_state(env):
    """Fruit centers [m], filtered pedicel forces [N], and detached flags."""
    return env.scene["plant"].observation()


@configclass
class TomatoObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        actions = ObservationTermCfg(func=mdp.last_action)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        tomatoes = ObservationTermCfg(func=fruit_state)
        concatenate_terms = False
        enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TomatoEventsCfg:
    reset_robot = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class TomatoTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class TomatoPickEnvCfg(FrankaCubeStackEnvCfg):
    """Pick the red fruit, pull until its pedicel breaks, then release on the tray.

    This interactive asset adapter intentionally supports one environment. Tissue
    stiffness and breaking loads are documented priors, not cultivar measurements.
    """

    def __post_init__(self):
        super().__post_init__()
        assets = Path(os.environ.get("ISAACLAB_TOMATO_ASSET_ROOT", DEFAULT_ASSET_ROOT)).expanduser()
        self.scene.num_envs = 1
        self.scene.cube_1 = self.scene.cube_2 = self.scene.cube_3 = None
        self.scene.plane.spawn = sim_utils.CuboidCfg(
            size=(30.0, 30.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.12, 0.08), roughness=1.0),
        )
        self.scene.plane.init_state.pos = (0.0, 0.0, -1.06)
        init = self.scene.robot.init_state
        init.joint_pos = {
            "panda_joint1": 0.86344563,
            "panda_joint2": -1.10514513,
            "panda_joint3": -1.0229305,
            "panda_joint4": -2.80593966,
            "panda_joint5": -2.88727748,
            "panda_joint6": 2.51838363,
            "panda_joint7": 1.2675895,
            "panda_finger_joint.*": 0.04,
        }
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=init)
        self.scene.robot.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)
        self.scene.robot.actuators["panda_hand"].stiffness = 400.0
        self.scene.robot.actuators["panda_hand"].damping = 20.0
        self.scene.robot.actuators["panda_hand"].joint_effort_limit = 10.0
        self.scene.plant = TomatoPlantCfg(
            prim_path="{ENV_REGEX_NS}/TomatoPlant",
            package_path=str(assets / "plant"),
            wind_velocity=(0.8, 0.0, 0.0),
            spawn=sim_utils.UsdFileCfg(usd_path=str(assets / "plant/plant_visual.usdz")),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.0, 0.0, -0.70710678, 0.70710678)),
        )
        self.scene.tray = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/HarvestTray",
            spawn=sim_utils.CuboidCfg(
                size=(0.22, 0.18, 0.015),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.14, 0.3, 0.15), roughness=0.65),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.36, -0.31, 0.009)),
        )
        background_mode = os.environ.get("ISAACLAB_TOMATO_BACKGROUND", "modeled")
        if background_mode not in {"0", "1", "modeled", "splats"}:
            raise ValueError("ISAACLAB_TOMATO_BACKGROUND must be modeled, splats, or 0")
        background = assets / (
            "greenhouse/greenhouse.usdz" if background_mode == "splats" else "greenhouse_modeled/greenhouse.usdz"
        )
        if background.is_file() and background_mode != "0":
            self.scene.background = AssetBaseCfg(
                prim_path="/World/Greenhouse", spawn=sim_utils.UsdFileCfg(usd_path=str(background))
            )
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            class_type=TomatoIKAction,
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        )
        self.actions.gripper_action.open_command_expr = {"panda_finger_.*": 0.04}
        self.gripper_open_val = 0.04
        self.observations = TomatoObservationsCfg()
        self.events = TomatoEventsCfg()
        self.terminations = TomatoTerminationsCfg()
        self.decimation = 4
        self.episode_length_s = 300.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.sim.default_visualizer_cfg = NewtonRTXVisualizerCfg(
            rtx_environment="none" if background_mode in {"1", "modeled"} else "studio",
            window_width=1280,
            window_height=720,
            eye=(-0.55, -1.0, 0.6),
            lookat=(0.44, 0.0, 0.3),
        )
        self.sim.physics = NewtonCfg(
            num_substeps=10,
            use_cuda_graph=False,
            solver_cfg=TomatoSolverCfg(
                class_type="isaaclab_tasks.contrib.tomato_pick.physics:NewtonTomatoManager",
                use_mujoco_contacts=True,
                use_mujoco_cpu=True,
                integrator="implicitfast",
                cone="elliptic",
                iterations=80,
                njmax=4096,
                nconmax=1024,
                update_data_interval=0,
            ),
        )
        self.sim.physics.default_shape_cfg.gap = 0.0
        self.viewer.eye = (-0.7, -1.25, 0.7)
        self.viewer.lookat = (0.48, 0.0, 0.28)
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    class_type=TomatoKeyboard, pos_sensitivity=0.006, rot_sensitivity=0.02, sim_device=self.sim.device
                ),
                "gamepad": TomatoGamepadCfg(sim_device=self.sim.device),
                "spacemouse": Se3SpaceMouseCfg(pos_sensitivity=0.006, rot_sensitivity=0.02, sim_device=self.sim.device),
            }
        )
