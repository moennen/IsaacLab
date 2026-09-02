# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
import os
from pathlib import Path

from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    HydroelasticSDFCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg, NewtonMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import MeshCollisionBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_contrib.custom_coupling import CoupledFeatherstoneVBDSolverCfg
from isaaclab_contrib.deformable.gaussian_twin import GaussianTwinDeformableObject, GaussianTwinRigidUsdFileCfg

from isaaclab_tasks.contrib.stack.stack_env_cfg import PhysicsCfg, mdp
from isaaclab_tasks.utils import PresetCfg

from . import stack_joint_pos_env_cfg

_LOGGER = logging.getLogger(__name__)

# Match newton.examples.mujoco_vbd_gaussian_twin simulation presets. Isaac Lab
# advances one 1/120 s physics tick at a time, whereas the example renders at
# 60 Hz, so its per-frame substep counts are halved here.
_BALANCED_GAUSSIAN_TWIN_SUBSTEPS = 4
_BALANCED_GAUSSIAN_TWIN_VBD_ITERATIONS = 45
_BALANCED_GAUSSIAN_TWIN_DRIVE_FREQUENCY_RATIO = 12.5 / 25.0
_FAST_GAUSSIAN_TWIN_SUBSTEPS = 2
_FAST_GAUSSIAN_TWIN_VBD_ITERATIONS = 30
_FAST_GAUSSIAN_TWIN_DRIVE_FREQUENCY_RATIO = 7.5 / 25.0

# Match the policy-training VBD presets from
# Isaac-Dexsuite-Deformable-Kuka-Allegro-Lift-v0.  These intentionally use a
# kinematic robot and particle-only rigid/soft contacts: their purpose is a
# stable, low-cost learning baseline rather than two-way force fidelity.
_DEXSUITE_SIMULATION_PROFILES = {
    "stable-kinematic": (8, 12, 0.75),
    "fast-kinematic": (4, 7, 1.0),
}
_DEXSUITE_TET_K_DAMP = 1.0e-5
_DEXSUITE_SOFT_CONTACT_KE = 8.0e3
_DEXSUITE_SHAPE_CONTACT_KE = 3.0e4
# Kept separate from tet elasticity damping even though the published
# DexSuite training preset uses the same numeric value for both quantities.
# Contact damping is a rigid/soft material parameter [N*s/m], not a tet
# material parameter.
_DEXSUITE_CONTACT_KD = 1.0e-5

# Virtual-proxy coupling trade-offs.  A large proxy inertia is appropriate for
# the stiff, ground-anchored Franka: it preconditions the coupled solve without
# moving its converged fixed point.  The demonstration profile deliberately
# applies its feedback slowly, prioritizing stable, inexpensive motion over a
# frame-exact robot reaction.
_GAUSSIAN_TWIN_PROXY_PROFILES = {
    "demo": (1, 1.0e3, 0.1),
    "balanced": (2, 1.0e3, 0.35),
    "accurate": (4, 1.0e3, 1.0),
}


def _optional_positive_int_env(name: str, default: int) -> int:
    """Read a positive integer environment override, treating an empty value as unset."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer, got '{value}'.") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer, got '{value}'.")
    return parsed


def _optional_nonnegative_float_env(name: str, default: float, *, positive: bool = False) -> float:
    """Read a finite nonnegative (or positive) floating-point environment override."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite number, got '{value}'.") from error
    if not math.isfinite(parsed) or (parsed <= 0.0 if positive else parsed < 0.0):
        comparator = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {comparator} number, got '{value}'.")
    return parsed


def _gaussian_twin_slot_position(slot_index: int, num_slots: int) -> tuple[float, float, float]:
    """Return a deterministic, non-overlapping table position for a toy slot."""
    # Preserve the single-object demonstration pose exactly.  Multiple toys
    # are laid out in pairs across the table, with enough clearance for the
    # largest packaged toy (about 0.28 m across) before physics starts.
    if num_slots == 1:
        return (0.5, 0.0, 0.05)
    column = slot_index % 2
    row = slot_index // 2
    return (0.42 + 0.22 * row, -0.24 + 0.48 * column, 0.05)


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
        if translation is not None and translation != (0.0, 0.0, 0.0):
            if translate_op is None:
                translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(*translation))
        if orientation is not None and orientation != (0.0, 0.0, 0.0, 1.0):
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

        background_disabled = os.environ.get("ISAACLAB_DISABLE_ALIGNED_BACKGROUND", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        aligned_background_value = os.environ.get("ISAACLAB_ALIGNED_BACKGROUND_USD")
        aligned_background = Path(aligned_background_value).expanduser() if aligned_background_value else None
        if not background_disabled and aligned_background is not None and aligned_background.is_file():
            self.scene.background = AssetBaseCfg(
                prim_path="/World/GaussianBackground",
                spawn=UsdFileCfg(
                    func=_spawn_aligned_background,
                    usd_path=str(aligned_background),
                ),
            )
        elif background_disabled:
            _LOGGER.info("Aligned Gaussian background disabled by ISAACLAB_DISABLE_ALIGNED_BACKGROUND.")
        elif aligned_background is None:
            _LOGGER.info(
                "No aligned Gaussian background configured; set ISAACLAB_ALIGNED_BACKGROUND_USD to enable one."
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
        self.scene.robot.spawn.rigid_props = sim_utils.MujocoRigidBodyPropertiesCfg(
            disable_gravity=False,
            gravcomp=1.0,
        )
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
class GaussianTwinCfg(PresetCfg):
    """USD-backed TetMesh and Gaussian-splat twin used by the Newton example."""

    newton_mjwarp_vbd_proxy: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/GaussianTwin",
        class_type=GaussianTwinDeformableObject,
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
        spawn=UsdFileCfg(
            usd_path=os.environ.get("ISAACLAB_GAUSSIAN_TWIN_ASSET", ""),
            make_uninstanceable=True,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            physics_material=NewtonDeformableBodyMaterialCfg(
                # GaussianTwinDeformableObject replaces these representative
                # values with per-package scale-aware material parameters.
                density=300.0,
                k_mu=4.1e4,
                k_lambda=9.6e4,
                k_damp=5.0,
                particle_radius=0.004,
            ),
        ),
    )
    default = newton_mjwarp_vbd_proxy


@configclass
class GaussianTwinRigidCfg(PresetCfg):
    """Rigid-body benchmark form of a packaged Gaussian twin."""

    newton_mjwarp: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GaussianTwin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
        spawn=GaussianTwinRigidUsdFileCfg(
            usd_path=os.environ.get("ISAACLAB_GAUSSIAN_TWIN_ASSET", ""),
            make_uninstanceable=True,
        ),
    )
    default = newton_mjwarp


@configclass
class GaussianTwinPhysicsCfg(PhysicsCfg):
    """Newton proxy coupling required for rigid Franka contact with the TetMesh."""

    newton_mjwarp_vbd_proxy = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        solver="newton",
                        integrator="implicitfast",
                        # MuJoCo remains responsible for rigid--rigid contacts
                        # (robot/table/ground).  The proxy pipeline below owns
                        # rigid--soft contact with the TetMesh.
                        use_mujoco_contacts=True,
                        njmax=600,
                        nconmax=400,
                        ls_iterations=20,
                        iterations=100,
                        ccd_iterations=35,
                    ),
                    bodies=[r"/World/envs/env_[^/]+/Robot"],
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=VBDSolverCfg(
                        iterations=60,
                        rigid_compliant_alm=True,
                        # Full-surface contacts can produce several records per
                        # TetMesh vertex while a finger closes.  Size the list
                        # for the packaged toys so contact rows are not dropped.
                        rigid_body_particle_contact_buffer_size=8192,
                    ),
                    all_particles=True,
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    # Only the gripper needs to be represented in the VBD
                    # view.  The proxy solver streams its pose from the full
                    # MJWarp Franka before each coupled solve.
                    bodies=[
                        # The stack task uses the legacy Franka USD (without
                        # ``Geometry``), while the Menagerie task configs use
                        # a ``Geometry`` scope.  The importer preserves both
                        # forms in body labels, so accept either layout.
                        r"/World/envs/env_[^/]+/Robot(?:/Geometry)?/.*panda_hand",
                        r"/World/envs/env_[^/]+/Robot(?:/Geometry)?/.*panda_(left|right)finger",
                    ],
                    collide_interval=1,
                    # Particle-only contacts let narrow fingers pass through a
                    # coarse TetMesh between vertices.
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        enable_rigid_soft_full_surface_contact=True,
                    ),
                )
            ],
            iterations=1,
        ),
        # Full-surface contact samples every participating mesh/convex SDF.
        # Provision one for importer-added Franka collision shapes too; the
        # task otherwise fails during CollisionPipeline construction as soon
        # as it encounters an unprovisioned link mesh.
        default_shape_cfg=NewtonShapeCfg(force_sdf=True),
        # The Newton example derives this from the tet-mesh wave speed.  Use a conservative
        # fixed value for the six random assets so the smallest elements are resolved too.
        num_substeps=32,
    )
    newton_dexsuite_kinematic = NewtonCfg(
        solver_cfg=CoupledFeatherstoneVBDSolverCfg(
            rigid_solver_cfg=FeatherstoneSolverCfg(update_mass_matrix_interval=8),
            soft_solver_cfg=VBDSolverCfg(
                iterations=12,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
                particle_collision_detection_interval=-1,
            ),
            kinematic_velocity_limit_scale=0.75,
            shape_material_ke=_DEXSUITE_SHAPE_CONTACT_KE,
            shape_material_kd=_DEXSUITE_CONTACT_KD,
            shape_material_mu=4.0,
        ),
        # DexSuite trains with vertex-sphere contacts.  In particular, do not
        # request SDF/full-surface contact in this mode: it is a qualitatively
        # different and much more expensive contact problem.
        collision_cfg=NewtonCollisionPipelineCfg(),
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=_DEXSUITE_SOFT_CONTACT_KE,
            soft_contact_kd=_DEXSUITE_CONTACT_KD,
            soft_contact_mu=4.0,
        ),
        default_shape_cfg=NewtonShapeCfg(
            ke=_DEXSUITE_SHAPE_CONTACT_KE,
            kd=_DEXSUITE_CONTACT_KD,
            mu=4.0,
        ),
        num_substeps=8,
    )
    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaGaussianTwinStackNewtonEnvCfg(FrankaCubeStackNewtonEnvCfg):
    """Franka IK task with reset-randomized Gaussian/TetMesh toy slots.

    The demonstration default preloads one package.  A Gaussian field is much
    larger than its TetMesh, so preloading all packages merely to choose one at
    reset makes RTX spend most of each frame processing invisible splats.
    """

    num_objects: int = 1
    """Number of randomly selected toy slots active per environment."""

    num_slots: int = 1
    """Number of toy packages preloaded from the asset directory."""

    def __post_init__(self):
        super().__post_init__()
        balanced_simulation = os.environ.get("ISAACLAB_GAUSSIAN_TWIN_BALANCED_SIMULATION", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        fast_simulation = os.environ.get("ISAACLAB_GAUSSIAN_TWIN_FAST_SIMULATION", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        dexsuite_simulation = os.environ.get("ISAACLAB_GAUSSIAN_TWIN_DEXSUITE_SIMULATION", "").strip().lower()
        rigid_simulation = os.environ.get("ISAACLAB_GAUSSIAN_TWIN_RIGID_SIMULATION", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if dexsuite_simulation and dexsuite_simulation not in _DEXSUITE_SIMULATION_PROFILES:
            choices = ", ".join(_DEXSUITE_SIMULATION_PROFILES)
            raise ValueError(
                f"ISAACLAB_GAUSSIAN_TWIN_DEXSUITE_SIMULATION must be one of {choices}, got '{dexsuite_simulation}'."
            )
        if sum((balanced_simulation, fast_simulation, bool(dexsuite_simulation), rigid_simulation)) > 1:
            raise ValueError(
                "ISAACLAB_GAUSSIAN_TWIN_BALANCED_SIMULATION, "
                "ISAACLAB_GAUSSIAN_TWIN_FAST_SIMULATION, "
                "ISAACLAB_GAUSSIAN_TWIN_DEXSUITE_SIMULATION, and "
                "ISAACLAB_GAUSSIAN_TWIN_RIGID_SIMULATION are mutually exclusive."
            )
        configured_num_objects = os.environ.get("ISAACLAB_GAUSSIAN_TWIN_NUM_OBJECTS")
        if configured_num_objects is not None:
            self.num_objects = int(configured_num_objects)
        configured_num_slots = os.environ.get("ISAACLAB_GAUSSIAN_TWIN_NUM_SLOTS")
        if configured_num_slots is not None:
            self.num_slots = int(configured_num_slots)
        # The aligned background is itself a high-density Gaussian capture.
        # It is useful for a composed recording but obscures the small robot
        # scene and dominates RTX frame time during interactive teleoperation.
        # Keep it opt-in for this task.
        if os.environ.get("ISAACLAB_ENABLE_ALIGNED_BACKGROUND", "").lower() not in {"1", "true", "yes", "on"}:
            self.scene.background = None
        self.scene.cube_1 = None
        self.scene.cube_2 = None
        self.scene.cube_3 = None
        asset_path = os.environ.get("ISAACLAB_GAUSSIAN_TWIN_ASSET")
        if asset_path:
            asset_paths = [Path(asset_path).expanduser()]
            if not asset_paths[0].is_file():
                raise ValueError(f"Gaussian twin asset does not exist: '{asset_paths[0]}'.")
            self.num_slots = 1
        else:
            asset_dir_value = os.environ.get("ISAACLAB_GAUSSIAN_TWIN_DIR")
            if not asset_dir_value:
                raise ValueError(
                    "No Gaussian twin asset configured. Set ISAACLAB_GAUSSIAN_TWIN_ASSET to one package or "
                    "ISAACLAB_GAUSSIAN_TWIN_DIR to a directory containing packaged Gaussian-twin USDA files."
                )
            asset_dir = Path(asset_dir_value).expanduser()
            if not asset_dir.is_dir():
                raise ValueError(f"Gaussian twin asset directory does not exist: '{asset_dir}'.")
            # The original Gaussian-twin packager emits ``*_package.usda``;
            # the DexSuite 4 mm / 512-node pipeline emits the equally valid
            # ``*_skinned_vbd_tet.usda`` form.  Both contain the same TetMesh
            # and baked Gaussian skinning contract consumed below.
            asset_paths = sorted(
                {*asset_dir.glob("baked.*_package.usda"), *asset_dir.glob("baked.*_skinned_vbd_tet.usda")}
            )
        required_assets = self.num_objects if rigid_simulation else self.num_slots
        if len(asset_paths) < required_assets:
            raise ValueError(
                f"Expected at least {required_assets} Gaussian twin packages, found {len(asset_paths)}. "
                "Set ISAACLAB_GAUSSIAN_TWIN_ASSET to one package or ISAACLAB_GAUSSIAN_TWIN_DIR to a package directory."
            )
        max_objects = len(asset_paths) if rigid_simulation else self.num_slots
        if not 1 <= self.num_objects <= max_objects:
            raise ValueError(f"num_objects must be in [1, {max_objects}], got {self.num_objects}.")
        splat_deformation = os.environ.get("ISAACLAB_GAUSSIAN_TWIN_SPLAT_DEFORMATION", "position-rotation-scale")
        valid_splat_deformations = {"position", "position-rotation", "position-rotation-scale"}
        if splat_deformation not in valid_splat_deformations:
            raise ValueError(
                "ISAACLAB_GAUSSIAN_TWIN_SPLAT_DEFORMATION must be one of "
                f"{', '.join(sorted(valid_splat_deformations))}, got '{splat_deformation}'."
            )
        if rigid_simulation:
            rigid_mass = _optional_nonnegative_float_env("ISAACLAB_GAUSSIAN_TWIN_RIGID_MASS", 0.05, positive=True)
            base_cfg = GaussianTwinRigidCfg().newton_mjwarp
            # There is no particle state to reset-select in rigid mode. Spawn
            # the requested number of packages deterministically instead.
            selected_paths = asset_paths[: self.num_objects]
            position_count = self.num_objects
        else:
            rigid_mass = None
            base_cfg = GaussianTwinCfg().newton_mjwarp_vbd_proxy
            selected_paths = asset_paths[: self.num_slots]
            position_count = self.num_slots
        for slot_index, asset_path in enumerate(selected_paths):
            slot_cfg = base_cfg.replace(
                prim_path=f"{{ENV_REGEX_NS}}/GaussianTwin_{slot_index}",
                spawn=base_cfg.spawn.replace(usd_path=str(asset_path)),
            )
            slot_cfg.init_state.pos = _gaussian_twin_slot_position(slot_index, position_count)
            if rigid_simulation:
                slot_cfg.spawn.rigid_mass = rigid_mass
            else:
                slot_cfg.slot_index = slot_index
                slot_cfg.num_objects = self.num_objects
                slot_cfg.num_slots = self.num_slots
                slot_cfg.splat_deformation = splat_deformation
            setattr(self.scene, f"gaussian_twin_{slot_index}", slot_cfg)
        # Newton can report a rank-deficient first-step Jacobian for the legacy stack robot.
        # Use the rank-aware SVD IK solver instead of DLS' explicit matrix inverse.
        self.actions.arm_action.controller = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="svd",
            ik_params={"min_singular_value": 1e-4},
        )
        if rigid_simulation:
            # ``super().__post_init__`` already installed the task's standard
            # MJWarp rigid-body configuration. Keep it intact for a direct
            # apples-to-apples benchmark against the VBD/proxy variants.
            # Unlike deformable objects, rigid objects have no particle-reset
            # implementation. Restore only their root poses and velocities on
            # reset; resetting the whole scene here would run after the robot
            # joint-randomization event and undo that event's result.
            for slot_index in range(self.num_objects):
                setattr(
                    self.events,
                    f"reset_rigid_gaussian_twin_{slot_index}",
                    EventTerm(
                        func=mdp.reset_root_state_uniform,
                        mode="reset",
                        params={
                            "pose_range": {},
                            "velocity_range": {},
                            "asset_cfg": SceneEntityCfg(f"gaussian_twin_{slot_index}"),
                        },
                    ),
                )
            _LOGGER.info(
                "Enabled Gaussian twin rigid benchmark: standard MJWarp with %d fixed package(s), %.3f kg each.",
                self.num_objects,
                rigid_mass,
            )
        elif dexsuite_simulation:
            substeps, iterations, velocity_limit_scale = _DEXSUITE_SIMULATION_PROFILES[dexsuite_simulation]
            self.sim.physics = GaussianTwinPhysicsCfg().newton_dexsuite_kinematic
            solver_cfg = self.sim.physics.solver_cfg
            self.sim.physics.num_substeps = substeps
            solver_cfg.rigid_solver_cfg.update_mass_matrix_interval = substeps
            solver_cfg.soft_solver_cfg.iterations = iterations
            solver_cfg.kinematic_velocity_limit_scale = velocity_limit_scale
            _LOGGER.info(
                "Enabled Gaussian twin DexSuite %s simulation: kinematic Featherstone + VBD, "
                "%d substeps, %d VBD iterations, %.2f velocity-limit scale.",
                dexsuite_simulation,
                substeps,
                iterations,
                velocity_limit_scale,
            )
        else:
            self.sim.physics = GaussianTwinPhysicsCfg().newton_mjwarp_vbd_proxy
            coupling_profile = os.environ.get("ISAACLAB_GAUSSIAN_TWIN_COUPLING_PROFILE", "demo").strip().lower()
            try:
                proxy_iterations, proxy_mass_scale, proxy_relaxation = _GAUSSIAN_TWIN_PROXY_PROFILES[coupling_profile]
            except KeyError as error:
                choices = ", ".join(_GAUSSIAN_TWIN_PROXY_PROFILES)
                raise ValueError(
                    f"ISAACLAB_GAUSSIAN_TWIN_COUPLING_PROFILE must be one of {choices}, got '{coupling_profile}'."
                ) from error
            solver_cfg = self.sim.physics.solver_cfg
            proxy_cfg = solver_cfg.proxies[0]
            solver_cfg.iterations = _optional_positive_int_env(
                "ISAACLAB_GAUSSIAN_TWIN_PROXY_ITERATIONS", proxy_iterations
            )
            proxy_cfg.mass_scale = _optional_nonnegative_float_env(
                "ISAACLAB_GAUSSIAN_TWIN_PROXY_MASS_SCALE", proxy_mass_scale, positive=True
            )
            proxy_cfg.proxy_relaxation = _optional_nonnegative_float_env(
                "ISAACLAB_GAUSSIAN_TWIN_PROXY_RELAXATION", proxy_relaxation, positive=True
            )
            _LOGGER.info(
                "Enabled Gaussian twin %s proxy coupling: %d iteration(s), %.0f mass scale, %.2f feedback relaxation.",
                coupling_profile,
                solver_cfg.iterations,
                proxy_cfg.mass_scale,
                proxy_cfg.proxy_relaxation,
            )
        if self.num_objects > 1 and not rigid_simulation:
            # VBD disables particle--particle contact by default.  This leaves
            # independently imported TetMeshes ghosting through one another.
            # These are contact-detection radii in metres, sized for the
            # packaged toys' ~4 mm particles; the larger margin avoids misses
            # at the task's 1/120 s tick.
            soft_solver = (
                self.sim.physics.solver_cfg.soft_solver_cfg
                if dexsuite_simulation
                else self.sim.physics.solver_cfg.entries[1].solver_cfg
            )
            soft_solver.particle_enable_self_contact = True
            soft_solver.particle_self_contact_radius = 0.008
            soft_solver.particle_self_contact_margin = 0.012
            soft_solver.particle_vertex_contact_buffer_size = 64
            soft_solver.particle_edge_contact_buffer_size = 128
        if balanced_simulation or fast_simulation:
            if balanced_simulation:
                substeps = _BALANCED_GAUSSIAN_TWIN_SUBSTEPS
                iterations = _BALANCED_GAUSSIAN_TWIN_VBD_ITERATIONS
                drive_frequency_ratio = _BALANCED_GAUSSIAN_TWIN_DRIVE_FREQUENCY_RATIO
                drive_frequency = 12.5
                profile_name = "balanced"
            else:
                substeps = _FAST_GAUSSIAN_TWIN_SUBSTEPS
                iterations = _FAST_GAUSSIAN_TWIN_VBD_ITERATIONS
                drive_frequency_ratio = _FAST_GAUSSIAN_TWIN_DRIVE_FREQUENCY_RATIO
                drive_frequency = 7.5
                profile_name = "fast"
            # Both reference profiles retain the material and contact gains
            # that make the toy graspable, while reducing temporal/nonlinear
            # solve cost and lowering gripper bandwidth to match the coarser
            # position-drive resolution.
            self.sim.physics.num_substeps = substeps
            self.sim.physics.solver_cfg.entries[1].solver_cfg.iterations = iterations
            hand = self.scene.robot.actuators["panda_hand"]
            hand.stiffness *= drive_frequency_ratio**2
            hand.damping *= drive_frequency_ratio
            _LOGGER.info(
                "Enabled Gaussian twin %s simulation: %d substeps, %d VBD iterations, %.1f Hz-equivalent hand drive.",
                profile_name,
                substeps,
                iterations,
                drive_frequency,
            )
        self.terminations.cube_1_dropping = None
        self.terminations.cube_2_dropping = None
        self.terminations.cube_3_dropping = None
        self.terminations.success = None
        self.events.randomize_cube_positions = None
        self.observations.policy.object = None
        self.observations.policy.cube_positions = None
        self.observations.policy.cube_orientations = None
        self.observations.subtask_terms = None


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
