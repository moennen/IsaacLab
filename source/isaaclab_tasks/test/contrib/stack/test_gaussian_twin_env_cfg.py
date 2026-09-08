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
    "ISAACLAB_GAUSSIAN_TWIN_COUPLING_PROFILE",
    "ISAACLAB_GAUSSIAN_TWIN_PROXY_ITERATIONS",
    "ISAACLAB_GAUSSIAN_TWIN_PROXY_MASS_SCALE",
    "ISAACLAB_GAUSSIAN_TWIN_PROXY_RELAXATION",
    "ISAACLAB_GAUSSIAN_TWIN_NUM_OBJECTS",
    "ISAACLAB_GAUSSIAN_TWIN_NUM_SLOTS",
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
    assert physics.soft_contact_cfg.soft_contact_mu == pytest.approx(10.0)
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


def test_rigid_benchmark_keeps_parent_robot_material(monkeypatch, tmp_path):
    """The VBD-specific contact tuning must not alter the rigid comparison mode."""
    cfg = _make_cfg(monkeypatch, tmp_path, ISAACLAB_GAUSSIAN_TWIN_RIGID_SIMULATION="1")

    newton_material = cfg.scene.robot.spawn.physics_material[1]
    assert newton_material.contact_stiffness == pytest.approx(1.0e6)
    assert cfg.scene.robot.actuators["panda_hand"].joint_effort_limit == pytest.approx(35.0)
