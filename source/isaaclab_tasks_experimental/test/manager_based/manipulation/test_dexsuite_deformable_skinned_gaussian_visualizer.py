# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import subprocess
import sys

import numpy as np
import warp as wp
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.dexsuite_deformable_env_cfg import (
    KIT_VIEW_MAX_GAUSSIANS_PER_ENV,
    TASK_VIEW_EYE,
    TASK_VIEW_LOOKAT,
)
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.skinned_gaussian_visualizer import (
    DEFAULT_SKINNED_GAUSSIAN_USD_PATH,
    SkinnedGaussianKitVisualizerCfg,
    SkinnedGaussianNewtonVisualizerCfg,
    _set_gaussian_casts_shadows,
    load_skinned_gaussian_visual_data,
    skin_gaussian_points_env_local_kernel,
    skin_gaussian_points_kernel,
)
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.tools.package_skinned_gaussian_tet_asset import (  # noqa: E501
    package_skinned_gaussian_tet_asset,
)
from isaaclab_visualizers.kit import KitVisualizerCfg

from pxr import Gf, Sdf, Usd

from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK_NAME = "Isaac-Dexsuite-Deformable-Kuka-Allegro-Lift-v0"
KIT_PLAY_TASK_NAME = "Isaac-Dexsuite-Deformable-Kuka-Allegro-Lift-Kit-Play-v0"


def _load_env_cfg():
    return load_cfg_from_registry(TASK_NAME, "env_cfg_entry_point")


def test_kit_play_cfg_import_does_not_preload_newton_or_pxr():
    code = f"""
import sys

import isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

load_cfg_from_registry({KIT_PLAY_TASK_NAME!r}, "env_cfg_entry_point")

if any(name == "pxr" or name.startswith("pxr.") for name in sys.modules):
    raise SystemExit("pxr was imported while loading the Kit play env cfg")
if any(name == "newton" or name.startswith("newton.") for name in sys.modules):
    raise SystemExit("newton was imported while loading the Kit play env cfg")
"""
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def _write_tiny_skinned_asset(tmp_path):
    gaussian_path = tmp_path / "gaussian.usda"
    tet_path = tmp_path / "tet.usda"
    output_path = tmp_path / "combined.usda"

    gaussian_stage = Usd.Stage.CreateNew(str(gaussian_path))
    gaussian_prim = gaussian_stage.DefinePrim("/World/Gaussians/gaussians_0", "ParticleField3DGaussianSplat")
    gaussian_prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set(
        [Gf.Vec3f(0.25, 0.25, 0.25), Gf.Vec3f(0.50, 0.25, 0.125)]
    )
    gaussian_prim.CreateAttribute("scales", Sdf.ValueTypeNames.Float3Array).Set(
        [Gf.Vec3f(0.001, 0.002, 0.003), Gf.Vec3f(0.004, 0.005, 0.006)]
    )
    gaussian_prim.CreateAttribute("radiance:sphericalHarmonicsCoefficients", Sdf.ValueTypeNames.Color3fArray).Set(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 1.0, 1.0),
            Gf.Vec3f(0.5, 0.0, -0.5),
            Gf.Vec3f(1.0, 1.0, 1.0),
        ]
    )
    gaussian_stage.Save()

    tet_stage = Usd.Stage.CreateNew(str(tet_path))
    tet_prim = tet_stage.DefinePrim("/TetMesh", "Xform")
    tet_prim.CreateAttribute("vbd:vertices", Sdf.ValueTypeNames.Point3fArray).Set(
        [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(0.0, 0.0, 1.0)]
    )
    tet_prim.CreateAttribute("vbd:tet_indices", Sdf.ValueTypeNames.IntArray).Set([0, 1, 2, 3])
    tet_stage.Save()

    package_skinned_gaussian_tet_asset(
        gaussian_usd_path=str(gaussian_path),
        tet_usd_path=str(tet_path),
        output_usd_path=str(output_path),
        rotate_tet_y_up_to_z_up=False,
        center_tet_to_origin=False,
        chunk_size=16,
    )
    return output_path


def test_skinned_gaussian_visualizer_is_disabled_by_default():
    env_cfg = resolve_presets(_load_env_cfg(), frozenset())

    assert env_cfg.sim.visualizer_cfgs == []


def test_skinned_gaussian_visualizer_preset_installs_task_newton_visualizer():
    env_cfg = resolve_presets(_load_env_cfg(), {"skinned_gaussian_visualizer"})

    assert isinstance(env_cfg.sim.visualizer_cfgs, SkinnedGaussianNewtonVisualizerCfg)
    assert env_cfg.sim.visualizer_cfgs.visualizer_type == "newton"
    assert env_cfg.sim.visualizer_cfgs.skinned_gaussian_usd_path == DEFAULT_SKINNED_GAUSSIAN_USD_PATH
    assert env_cfg.sim.visualizer_cfgs.max_visible_envs == 1
    assert env_cfg.sim.visualizer_cfgs.show_tet_surface is False
    assert env_cfg.sim.visualizer_cfgs.show_tet_particles is False


def test_kit_visualizer_preset_installs_task_kit_visualizer():
    env_cfg = resolve_presets(_load_env_cfg(), {"kit_visualizer"})

    assert isinstance(env_cfg.sim.visualizer_cfgs, list)
    assert len(env_cfg.sim.visualizer_cfgs) == 2
    gaussian_overlay, kit_visualizer = env_cfg.sim.visualizer_cfgs
    assert isinstance(gaussian_overlay, SkinnedGaussianKitVisualizerCfg)
    assert gaussian_overlay.visualizer_type == "kit"
    assert gaussian_overlay.max_gaussians_per_env == KIT_VIEW_MAX_GAUSSIANS_PER_ENV
    assert gaussian_overlay.hide_tet_visual_mesh is True
    assert isinstance(kit_visualizer, KitVisualizerCfg)
    assert kit_visualizer.visualizer_type == "kit"
    assert kit_visualizer.eye == TASK_VIEW_EYE
    assert kit_visualizer.lookat == TASK_VIEW_LOOKAT


def test_kit_play_env_uses_kit_visualizer_cfg():
    env_cfg = load_cfg_from_registry(KIT_PLAY_TASK_NAME, "env_cfg_entry_point")

    assert env_cfg.scene.num_envs == 16
    assert isinstance(env_cfg.sim.visualizer_cfgs, list)
    assert len(env_cfg.sim.visualizer_cfgs) == 2
    gaussian_overlay, kit_visualizer = env_cfg.sim.visualizer_cfgs
    assert isinstance(gaussian_overlay, SkinnedGaussianKitVisualizerCfg)
    assert gaussian_overlay.visualizer_type == "kit"
    assert gaussian_overlay.max_gaussians_per_env == KIT_VIEW_MAX_GAUSSIANS_PER_ENV
    assert gaussian_overlay.hide_tet_visual_mesh is True
    assert isinstance(kit_visualizer, KitVisualizerCfg)
    assert kit_visualizer.visualizer_type == "kit"
    assert env_cfg.viewer.eye == TASK_VIEW_EYE
    assert env_cfg.viewer.lookat == TASK_VIEW_LOOKAT
    assert kit_visualizer.eye == TASK_VIEW_EYE
    assert kit_visualizer.lookat == TASK_VIEW_LOOKAT


def test_load_skinned_gaussian_visual_data_reads_custom_binding(tmp_path):
    usd_path = _write_tiny_skinned_asset(tmp_path)

    data = load_skinned_gaussian_visual_data(str(usd_path), max_gaussians_per_env=None, radius_scale=2.0)

    assert data.source_count == 2
    assert data.selected_count == 2
    np.testing.assert_array_equal(data.selected_indices, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(data.influence_indices, np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_allclose(data.influence_weights.reshape(-1, 4)[0], [0.25, 0.25, 0.25, 0.25], atol=1.0e-7)
    np.testing.assert_allclose(data.radii, [0.004, 0.010], atol=1.0e-7)
    np.testing.assert_allclose(data.colors[0], [0.5, 0.5, 0.5], atol=1.0e-7)


def test_set_gaussian_casts_shadows_disables_do_not_cast_shadow_primvar(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "gaussian.usda"))
    gaussian_prim = stage.DefinePrim("/World/Gaussian", "ParticleField3DGaussianSplat")
    gaussian_prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)

    _set_gaussian_casts_shadows(gaussian_prim)

    attr = gaussian_prim.GetAttribute("primvars:doNotCastShadows")
    assert attr.IsValid()
    assert attr.GetTypeName() == Sdf.ValueTypeNames.Bool
    assert attr.Get() is False


def test_skin_gaussian_points_kernel_skins_visible_envs():
    particle_q = wp.array(
        [
            wp.vec3f(0.0, 0.0, 0.0),
            wp.vec3f(1.0, 0.0, 0.0),
            wp.vec3f(0.0, 1.0, 0.0),
            wp.vec3f(0.0, 0.0, 1.0),
            wp.vec3f(10.0, 0.0, 0.0),
            wp.vec3f(11.0, 0.0, 0.0),
            wp.vec3f(10.0, 1.0, 0.0),
            wp.vec3f(10.0, 0.0, 1.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    particle_offsets = wp.array([0, 4], dtype=wp.int32, device="cpu")
    visible_env_ids = wp.array([0, 1], dtype=wp.int32, device="cpu")
    influence_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device="cpu")
    influence_weights = wp.array([0.25, 0.25, 0.25, 0.25], dtype=wp.float32, device="cpu")
    out_points = wp.empty(2, dtype=wp.vec3f, device="cpu")

    wp.launch(
        skin_gaussian_points_kernel,
        dim=2,
        inputs=[particle_q, particle_offsets, visible_env_ids, influence_indices, influence_weights, 1],
        outputs=[out_points],
        device="cpu",
    )

    np.testing.assert_allclose(out_points.numpy(), [[0.25, 0.25, 0.25], [10.25, 0.25, 0.25]], atol=1.0e-7)


def test_skin_gaussian_points_env_local_kernel_subtracts_env_origins():
    particle_q = wp.array(
        [
            wp.vec3f(0.0, 0.0, 0.0),
            wp.vec3f(1.0, 0.0, 0.0),
            wp.vec3f(0.0, 1.0, 0.0),
            wp.vec3f(0.0, 0.0, 1.0),
            wp.vec3f(10.0, 0.0, 0.0),
            wp.vec3f(11.0, 0.0, 0.0),
            wp.vec3f(10.0, 1.0, 0.0),
            wp.vec3f(10.0, 0.0, 1.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    particle_offsets = wp.array([0, 4], dtype=wp.int32, device="cpu")
    visible_env_ids = wp.array([0, 1], dtype=wp.int32, device="cpu")
    env_position_offsets = wp.array(
        [wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(10.0, 0.0, 0.0)],
        dtype=wp.vec3f,
        device="cpu",
    )
    influence_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device="cpu")
    influence_weights = wp.array([0.25, 0.25, 0.25, 0.25], dtype=wp.float32, device="cpu")
    out_points = wp.empty(2, dtype=wp.vec3f, device="cpu")

    wp.launch(
        skin_gaussian_points_env_local_kernel,
        dim=2,
        inputs=[
            particle_q,
            particle_offsets,
            visible_env_ids,
            env_position_offsets,
            influence_indices,
            influence_weights,
            1,
        ],
        outputs=[out_points],
        device="cpu",
    )

    np.testing.assert_allclose(out_points.numpy(), [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]], atol=1.0e-7)
