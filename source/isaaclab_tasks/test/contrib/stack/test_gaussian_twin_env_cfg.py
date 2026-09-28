# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free tests for the Gaussian-twin Newton physics profiles."""

import pytest
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

from isaaclab_tasks.contrib.stack.config.franka.stack_ik_rel_env_cfg import (
    FrankaGaussianTwinStackNewtonEnvCfg,
)

_GAUSSIAN_TWIN_ENV_VARS = (
    "ISAACLAB_GAUSSIAN_TWIN_ASSET",
    "ISAACLAB_GAUSSIAN_TWIN_DIR",
    "ISAACLAB_GAUSSIAN_TWIN_BALANCED_SIMULATION",
    "ISAACLAB_GAUSSIAN_TWIN_GRASP_FAST_SIMULATION",
    "ISAACLAB_GAUSSIAN_TWIN_FAST_SIMULATION",
    "ISAACLAB_GAUSSIAN_TWIN_DEXSUITE_SIMULATION",
    "ISAACLAB_GAUSSIAN_TWIN_SIMPLICITS_SIMULATION",
    "ISAACLAB_GAUSSIAN_TWIN_RIGID_SIMULATION",
    "ISAACLAB_GAUSSIAN_TWIN_FULL_SURFACE_CONTACT",
    "ISAACLAB_GAUSSIAN_TWIN_CONTACT_FRICTION",
    "ISAACLAB_GAUSSIAN_TWIN_GRIPPER_STIFFNESS",
    "ISAACLAB_GAUSSIAN_TWIN_GRIPPER_DAMPING",
    "ISAACLAB_GAUSSIAN_TWIN_GRIPPER_EFFORT_LIMIT",
    "ISAACLAB_GAUSSIAN_TWIN_COUPLING_PROFILE",
    "ISAACLAB_GAUSSIAN_TWIN_PROXY_ITERATIONS",
    "ISAACLAB_GAUSSIAN_TWIN_PROXY_MASS_SCALE",
    "ISAACLAB_GAUSSIAN_TWIN_PROXY_RELAXATION",
    "ISAACLAB_GAUSSIAN_TWIN_NUM_OBJECTS",
    "ISAACLAB_GAUSSIAN_TWIN_NUM_SLOTS",
    "ISAACLAB_ALIGNED_BACKGROUND_USD",
    "ISAACLAB_DISABLE_ALIGNED_BACKGROUND",
    "ISAACLAB_ENABLE_ALIGNED_BACKGROUND",
)


def _make_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path, **environment):
    for name in _GAUSSIAN_TWIN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    asset_path = tmp_path / "baked.test_package.usda"
    asset_path.touch()
    monkeypatch.setenv("ISAACLAB_GAUSSIAN_TWIN_ASSET", str(asset_path))
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return FrankaGaussianTwinStackNewtonEnvCfg()


def test_default_packages_resolve_from_shared_task_cache(monkeypatch, tmp_path):
    for name in _GAUSSIAN_TWIN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ISAACLAB_TASK_ASSET_ROOT", str(tmp_path))
    packages = tmp_path / "gaussian_twin/packages"
    packages.mkdir(parents=True)
    asset = packages / "baked.test_package.usda"
    asset.touch()
    cfg = FrankaGaussianTwinStackNewtonEnvCfg()
    assert cfg.scene.gaussian_twin_0.spawn.usd_path == str(asset)


@pytest.mark.parametrize("disabled", [False, True])
def test_ebc_background_defaults_to_task_cache_and_can_be_disabled(monkeypatch, tmp_path, disabled):
    background = tmp_path / "gaussian_twin/background/ebc/aligned.usda"
    background.parent.mkdir(parents=True)
    background.touch()
    cfg = _make_cfg(
        monkeypatch,
        tmp_path,
        ISAACLAB_TASK_ASSET_ROOT=str(tmp_path),
        ISAACLAB_DISABLE_ALIGNED_BACKGROUND="1" if disabled else "0",
        ISAACLAB_GAUSSIAN_TWIN_DIR=str(tmp_path / "separate-toy-packages"),
    )
    if disabled:
        assert getattr(cfg.scene, "background", None) is None
    else:
        assert cfg.scene.background.spawn.usd_path == str(background)
        assert cfg.sim.default_visualizer_cfg.eye == cfg.viewer.eye == (1.8, -2.2, 1.4)
        assert cfg.sim.default_visualizer_cfg.lookat == cfg.viewer.lookat == (0.35, 0.0, 0.1)


def test_explicit_background_override_is_preserved(monkeypatch, tmp_path):
    background = tmp_path / "custom.usda"
    background.touch()
    cfg = _make_cfg(monkeypatch, tmp_path, ISAACLAB_ALIGNED_BACKGROUND_USD=str(background))
    assert cfg.scene.background.spawn.usd_path == str(background)


def test_native_profile_uses_interactive_solver_and_soft_contact(monkeypatch, tmp_path):
    """Native mode must not inherit the rigid cube task's hydroelastic contact gains."""
    cfg = _make_cfg(monkeypatch, tmp_path)
    physics = cfg.sim.physics
    proxy = physics.solver_cfg.proxies[0]
    soft_solver = physics.solver_cfg.entries[1].solver_cfg

    assert physics.num_substeps == 4
    assert soft_solver.iterations == 10
    assert proxy.collision_pipeline.enable_rigid_soft_full_surface_contact is False
    assert physics.default_shape_cfg.force_sdf is False
    assert physics.soft_contact_cfg.soft_contact_mu == pytest.approx(2.0)
    assert cfg.scene.robot.actuators["panda_hand"].joint_effort_limit == pytest.approx(10.0)

    newton_material = cfg.scene.robot.spawn.physics_material[1]
    assert isinstance(newton_material, NewtonMaterialCfg)
    assert newton_material.contact_stiffness == pytest.approx(3.0e4)
    assert newton_material.contact_damping == pytest.approx(10.0)


def test_full_surface_contact_is_explicit_opt_in(monkeypatch, tmp_path):
    cfg = _make_cfg(monkeypatch, tmp_path, ISAACLAB_GAUSSIAN_TWIN_FULL_SURFACE_CONTACT="1")
    proxy = cfg.sim.physics.solver_cfg.proxies[0]

    assert proxy.collision_pipeline.enable_rigid_soft_full_surface_contact is True
    assert cfg.sim.physics.default_shape_cfg.force_sdf is True


def test_native_contact_friction_can_be_overridden(monkeypatch, tmp_path):
    cfg = _make_cfg(monkeypatch, tmp_path, ISAACLAB_GAUSSIAN_TWIN_CONTACT_FRICTION="4.0")

    assert cfg.sim.physics.soft_contact_cfg.soft_contact_mu == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("environment", "expected_substeps", "expected_vbd_iterations", "expected_mjwarp_iterations"),
    [
        ("ISAACLAB_GAUSSIAN_TWIN_FAST_SIMULATION", 2, 10, 100),
        ("ISAACLAB_GAUSSIAN_TWIN_GRASP_FAST_SIMULATION", 4, 8, 50),
        ("ISAACLAB_GAUSSIAN_TWIN_BALANCED_SIMULATION", 4, 15, 100),
    ],
)
def test_interactive_profiles_limit_vbd_work(
    monkeypatch,
    tmp_path,
    environment,
    expected_substeps,
    expected_vbd_iterations,
    expected_mjwarp_iterations,
):
    cfg = _make_cfg(monkeypatch, tmp_path, **{environment: "1"})

    assert cfg.sim.physics.num_substeps == expected_substeps
    assert cfg.sim.physics.solver_cfg.entries[1].solver_cfg.iterations == expected_vbd_iterations
    assert cfg.sim.physics.solver_cfg.entries[0].solver_cfg.iterations == expected_mjwarp_iterations


def test_grasp_fast_profile_uses_forceful_compliant_hand(monkeypatch, tmp_path):
    cfg = _make_cfg(monkeypatch, tmp_path, ISAACLAB_GAUSSIAN_TWIN_GRASP_FAST_SIMULATION="1")
    hand = cfg.scene.robot.actuators["panda_hand"]

    assert hand.stiffness == pytest.approx(600.0)
    assert hand.damping == pytest.approx(40.0)
    assert hand.joint_effort_limit == pytest.approx(20.0)


def test_grasp_fast_hand_can_be_overridden(monkeypatch, tmp_path):
    cfg = _make_cfg(
        monkeypatch,
        tmp_path,
        ISAACLAB_GAUSSIAN_TWIN_GRASP_FAST_SIMULATION="1",
        ISAACLAB_GAUSSIAN_TWIN_GRIPPER_STIFFNESS="750.0",
        ISAACLAB_GAUSSIAN_TWIN_GRIPPER_DAMPING="50.0",
        ISAACLAB_GAUSSIAN_TWIN_GRIPPER_EFFORT_LIMIT="25.0",
    )
    hand = cfg.scene.robot.actuators["panda_hand"]

    assert hand.stiffness == pytest.approx(750.0)
    assert hand.damping == pytest.approx(50.0)
    assert hand.joint_effort_limit == pytest.approx(25.0)


def test_rigid_benchmark_keeps_parent_robot_material(monkeypatch, tmp_path):
    """The VBD-specific contact tuning must not alter the rigid comparison mode."""
    cfg = _make_cfg(monkeypatch, tmp_path, ISAACLAB_GAUSSIAN_TWIN_RIGID_SIMULATION="1")

    newton_material = cfg.scene.robot.spawn.physics_material[1]
    assert newton_material.contact_stiffness == pytest.approx(1.0e6)
    assert cfg.scene.robot.actuators["panda_hand"].joint_effort_limit == pytest.approx(35.0)
