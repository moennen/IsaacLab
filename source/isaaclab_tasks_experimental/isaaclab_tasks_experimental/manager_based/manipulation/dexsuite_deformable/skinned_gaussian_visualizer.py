# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local Newton visualizer for Gaussian splats skinned to the deformable tet proxy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

from isaaclab.utils.configclass import configclass

logger = logging.getLogger(__name__)

DEFAULT_SKINNED_GAUSSIAN_USD_PATH = "/tmp/blueHairRagdoll_skinned_gaussian_tet.usdc"

_SH_C0 = 0.28209479177387814


@wp.kernel
def skin_gaussian_points_kernel(
    particle_q: wp.array(dtype=wp.vec3f),
    particle_offsets: wp.array(dtype=wp.int32),
    visible_env_ids: wp.array(dtype=wp.int32),
    influence_indices: wp.array(dtype=wp.int32),
    influence_weights: wp.array(dtype=wp.float32),
    gaussian_count: int,
    out_points: wp.array(dtype=wp.vec3f),
):
    """Skin selected Gaussian centers from tet particle positions."""
    tid = wp.tid()
    env_slot = tid // gaussian_count
    gaussian_slot = tid - env_slot * gaussian_count
    influence_offset = gaussian_slot * 4
    particle_offset = particle_offsets[visible_env_ids[env_slot]]

    i0 = particle_offset + influence_indices[influence_offset + 0]
    i1 = particle_offset + influence_indices[influence_offset + 1]
    i2 = particle_offset + influence_indices[influence_offset + 2]
    i3 = particle_offset + influence_indices[influence_offset + 3]

    w0 = influence_weights[influence_offset + 0]
    w1 = influence_weights[influence_offset + 1]
    w2 = influence_weights[influence_offset + 2]
    w3 = influence_weights[influence_offset + 3]

    out_points[tid] = particle_q[i0] * w0 + particle_q[i1] * w1 + particle_q[i2] * w2 + particle_q[i3] * w3


@dataclass(frozen=True)
class SkinnedGaussianVisualData:
    """CPU-side Gaussian skinning data loaded from USD."""

    influence_indices: np.ndarray
    influence_weights: np.ndarray
    radii: np.ndarray
    colors: np.ndarray
    source_count: int
    selected_count: int
    stride: int


@dataclass
class _SkinnedGaussianRuntime:
    """GPU buffers owned by the skinned Gaussian visualizer."""

    asset: object
    influence_indices: wp.array
    influence_weights: wp.array
    visible_env_ids: wp.array
    radii: wp.array
    colors: wp.array
    points: wp.array
    gaussian_count: int
    total_points: int
    colors_pending_upload: bool = True


def _find_first_gaussian_prim(stage):
    for prim in stage.Traverse():
        if prim.GetTypeName() == "ParticleField3DGaussianSplat":
            return prim
    raise ValueError(f"No ParticleField3DGaussianSplat prim found in '{stage.GetRootLayer().identifier}'.")


def _as_numpy_array(value, dtype: np.dtype, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.size == 0:
        raise ValueError(f"Attribute '{name}' is empty.")
    return np.ascontiguousarray(array)


def _selected_indices(count: int, max_count: int | None) -> tuple[np.ndarray, int]:
    if max_count is None or max_count <= 0 or count <= max_count:
        return np.arange(count, dtype=np.int32), 1
    stride = int(np.ceil(count / max_count))
    return np.arange(0, count, stride, dtype=np.int32)[:max_count], stride


def _colors_from_gaussian_prim(gaussian_prim, point_count: int, selected: np.ndarray) -> np.ndarray:
    sh_attr = gaussian_prim.GetAttribute("radiance:sphericalHarmonicsCoefficients")
    if sh_attr.IsValid() and sh_attr.HasValue():
        sh = _as_numpy_array(sh_attr.Get(), np.float32, name=sh_attr.GetName())
        if sh.ndim == 2 and sh.shape[1] == 3 and sh.shape[0] % point_count == 0:
            sh = sh.reshape(point_count, sh.shape[0] // point_count, 3)
            colors = (_SH_C0 * sh[selected, 0, :] + 0.5).clip(0.0, 1.0)
            return np.ascontiguousarray(colors, dtype=np.float32)

    return np.ascontiguousarray(np.tile(np.asarray((0.45, 0.55, 0.95), dtype=np.float32), (selected.size, 1)))


def load_skinned_gaussian_visual_data(
    usd_path: str,
    gaussian_prim_path: str | None = None,
    *,
    max_gaussians_per_env: int | None = 20_000,
    radius_scale: float = 4.0,
    min_radius: float = 0.001,
) -> SkinnedGaussianVisualData:
    """Load Gaussian-to-tet skinning metadata from a combined USD asset."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"Failed to open skinned Gaussian USD: '{usd_path}'.")

    gaussian_prim = stage.GetPrimAtPath(gaussian_prim_path) if gaussian_prim_path else _find_first_gaussian_prim(stage)
    if not gaussian_prim.IsValid():
        raise ValueError(f"Could not find Gaussian prim '{gaussian_prim_path}' in '{usd_path}'.")

    point_count_attr = gaussian_prim.GetAttribute("newton:deformableSkin:pointCount")
    influence_size_attr = gaussian_prim.GetAttribute("newton:deformableSkin:influenceSize")
    point_count = int(point_count_attr.Get()) if point_count_attr.IsValid() and point_count_attr.HasValue() else 0
    influence_size = (
        int(influence_size_attr.Get()) if influence_size_attr.IsValid() and influence_size_attr.HasValue() else 0
    )
    if point_count <= 0 or influence_size != 4:
        raise ValueError(
            f"Gaussian prim '{gaussian_prim.GetPath()}' does not define supported newton:deformableSkin metadata."
        )

    influence_indices_attr = gaussian_prim.GetAttribute("newton:deformableSkin:influenceIndices")
    influence_weights_attr = gaussian_prim.GetAttribute("newton:deformableSkin:influenceWeights")
    if not influence_indices_attr.IsValid() or not influence_weights_attr.IsValid():
        raise ValueError(f"Gaussian prim '{gaussian_prim.GetPath()}' is missing skinning influence arrays.")

    influence_indices = _as_numpy_array(influence_indices_attr.Get(), np.int32, name=influence_indices_attr.GetName())
    influence_weights = _as_numpy_array(influence_weights_attr.Get(), np.float32, name=influence_weights_attr.GetName())
    if influence_indices.size != point_count * 4 or influence_weights.size != point_count * 4:
        raise ValueError(
            "Skinning influence arrays must contain exactly four entries per Gaussian "
            f"(point_count={point_count}, indices={influence_indices.size}, weights={influence_weights.size})."
        )
    if np.any(influence_indices < 0):
        raise ValueError(f"Gaussian prim '{gaussian_prim.GetPath()}' contains negative skinning influence indices.")

    selected, stride = _selected_indices(point_count, max_gaussians_per_env)
    influence_indices = np.ascontiguousarray(influence_indices.reshape(point_count, 4)[selected].reshape(-1))
    influence_weights = np.ascontiguousarray(influence_weights.reshape(point_count, 4)[selected].reshape(-1))

    scales_attr = gaussian_prim.GetAttribute("scales")
    if scales_attr.IsValid() and scales_attr.HasValue():
        scales = _as_numpy_array(scales_attr.Get(), np.float32, name=scales_attr.GetName()).reshape(point_count, 3)
        radii = np.maximum(scales[selected].mean(axis=1) * float(radius_scale), float(min_radius))
    else:
        radii = np.full(selected.shape[0], float(min_radius), dtype=np.float32)

    colors = _colors_from_gaussian_prim(gaussian_prim, point_count, selected)

    return SkinnedGaussianVisualData(
        influence_indices=np.ascontiguousarray(influence_indices, dtype=np.int32),
        influence_weights=np.ascontiguousarray(influence_weights, dtype=np.float32),
        radii=np.ascontiguousarray(radii, dtype=np.float32),
        colors=np.ascontiguousarray(colors, dtype=np.float32),
        source_count=point_count,
        selected_count=int(selected.size),
        stride=stride,
    )


@configclass
class SkinnedGaussianNewtonVisualizerCfg(NewtonVisualizerCfg):
    """Newton visualizer that overlays skinned Gaussian positions for the deformable task."""

    skinned_gaussian_usd_path: str = DEFAULT_SKINNED_GAUSSIAN_USD_PATH
    """Combined Gaussian + tet USD containing ``newton:deformableSkin:*`` metadata."""

    gaussian_prim_path: str | None = None
    """Optional Gaussian prim path. When omitted, the first ParticleField3DGaussianSplat prim is used."""

    deformable_asset_name: str = "deformable"
    """Scene asset name of the deformable object whose tet particles drive the Gaussians."""

    max_gaussians_per_env: int | None = 20_000
    """Maximum rendered Gaussian points per visible environment. Non-positive means all points."""

    radius_scale: float = 4.0
    """Multiplier applied to Gaussian scale-derived sphere radii."""

    min_radius: float = 0.001
    """Minimum rendered sphere radius in meters."""

    point_cloud_name: str = "/task/skinned_gaussians"
    """Newton viewer object name used for the skinned Gaussian point cloud."""

    show_tet_surface: bool = False
    """Whether to show Newton's default deformable triangle surface in addition to the skinned Gaussians."""

    show_tet_particles: bool = False
    """Whether to show Newton's default deformable particles in addition to the skinned Gaussians."""

    max_visible_envs: int | None = 1
    randomly_sample_visible_envs: bool = False
    eye: tuple[float, float, float] = (-2.20, 0.10, 0.90)
    lookat: tuple[float, float, float] = (-0.55, 0.05, 0.45)

    def create_visualizer(self):
        """Create the task-specific Newton visualizer."""
        return _create_skinned_gaussian_newton_visualizer(self)


def _create_skinned_gaussian_newton_visualizer(cfg: SkinnedGaussianNewtonVisualizerCfg):
    from isaaclab_visualizers.newton.newton_visualizer import NewtonVisualizer

    class SkinnedGaussianNewtonVisualizer(_SkinnedGaussianNewtonVisualizerMixin, NewtonVisualizer):
        pass

    return SkinnedGaussianNewtonVisualizer(cfg)


class _SkinnedGaussianNewtonVisualizerMixin:
    """Newton visualizer overlaying skinned Gaussian proxy points."""

    cfg: SkinnedGaussianNewtonVisualizerCfg

    def __init__(self, cfg: SkinnedGaussianNewtonVisualizerCfg):
        super().__init__(cfg)
        self._skinned_gaussian: _SkinnedGaussianRuntime | None = None
        self._skinned_gaussian_load_error: str | None = None

    def initialize(self, scene_data_provider) -> None:
        """Initialize Newton visualizer and task-local skinned Gaussian buffers."""
        super().initialize(scene_data_provider)
        if self._viewer is not None:
            self._viewer.show_triangles = self.cfg.show_tet_surface
            self._viewer.show_particles = self.cfg.show_tet_particles
        self._initialize_skinned_gaussian_runtime()

    def step(self, dt: float) -> None:
        """Advance visualization and log the skinned Gaussian point cloud inside the Newton frame."""
        from isaaclab_newton.physics import NewtonManager

        if not self._is_initialized or self._is_closed:
            return

        self._sim_time += dt
        self._step_counter += 1

        if self._viewer is None:
            self._state = NewtonManager.get_state(self._scene_data_provider)
            return

        self._state = NewtonManager.get_state(self._scene_data_provider)

        update_frequency = self._viewer._update_frequency if self._viewer else self._update_frequency
        if self._step_counter % update_frequency != 0:
            return

        num_envs = NewtonManager.get_num_envs()

        try:
            if not self._viewer.is_paused():
                self._viewer.begin_frame(self._sim_time)
                try:
                    if self._state is not None:
                        body_q = getattr(self._state, "body_q", None)
                        if hasattr(body_q, "shape") and body_q.shape[0] == 0:
                            return
                        self._viewer.log_state(self._state)
                        self._log_skinned_gaussians()
                        if self.cfg.enable_markers:
                            from isaaclab_visualizers.newton.newton_visualization_markers import (
                                render_newton_visualization_markers,
                            )

                            render_newton_visualization_markers(
                                self._viewer, self._resolved_visible_env_ids, num_envs=num_envs
                            )
                        self._log_camera_sensor_image()
                finally:
                    self._viewer.end_frame()
            else:
                self._viewer._update()
        except Exception:
            logger.exception("[SkinnedGaussianNewtonVisualizer] Viewer update failed.")

    def _initialize_skinned_gaussian_runtime(self) -> None:
        if self._viewer is None or self._scene_data_provider is None:
            return

        usd_path = Path(self.cfg.skinned_gaussian_usd_path).expanduser()
        if not usd_path.is_file():
            self._skinned_gaussian_load_error = f"skinned Gaussian USD does not exist: '{usd_path}'"
            logger.warning("[SkinnedGaussianNewtonVisualizer] %s", self._skinned_gaussian_load_error)
            return

        scene = self._scene_data_provider.get_interactive_scene()
        if scene is None:
            self._skinned_gaussian_load_error = "interactive scene is unavailable"
            logger.warning("[SkinnedGaussianNewtonVisualizer] %s", self._skinned_gaussian_load_error)
            return
        try:
            asset = scene[self.cfg.deformable_asset_name]
        except KeyError:
            self._skinned_gaussian_load_error = (
                f"scene has no deformable asset named '{self.cfg.deformable_asset_name}'"
            )
            logger.warning("[SkinnedGaussianNewtonVisualizer] %s", self._skinned_gaussian_load_error)
            return

        particle_offsets = getattr(asset.data, "_particle_offsets", None)
        particles_per_body = getattr(asset.data, "_particles_per_body", None)
        if particle_offsets is None or particles_per_body is None:
            self._skinned_gaussian_load_error = "deformable asset does not expose Newton particle offsets"
            logger.warning("[SkinnedGaussianNewtonVisualizer] %s", self._skinned_gaussian_load_error)
            return

        try:
            visual_data = load_skinned_gaussian_visual_data(
                str(usd_path),
                self.cfg.gaussian_prim_path,
                max_gaussians_per_env=self.cfg.max_gaussians_per_env,
                radius_scale=self.cfg.radius_scale,
                min_radius=self.cfg.min_radius,
            )
        except Exception as exc:
            self._skinned_gaussian_load_error = str(exc)
            logger.warning("[SkinnedGaussianNewtonVisualizer] Failed to load skinned Gaussian data: %s", exc)
            return

        if int(visual_data.influence_indices.max(initial=0)) >= int(particles_per_body):
            self._skinned_gaussian_load_error = (
                f"skinning references tet vertex {int(visual_data.influence_indices.max())}, "
                f"but deformable asset has only {int(particles_per_body)} particles per body"
            )
            logger.warning("[SkinnedGaussianNewtonVisualizer] %s", self._skinned_gaussian_load_error)
            return

        num_envs = self._scene_data_provider.num_envs
        env_ids = self._resolved_visible_env_ids
        visible_env_ids = (
            np.arange(num_envs, dtype=np.int32) if env_ids is None else np.asarray(env_ids, dtype=np.int32)
        )
        if visible_env_ids.size == 0:
            logger.info("[SkinnedGaussianNewtonVisualizer] No visible envs selected; Gaussian overlay disabled.")
            return

        device = self._viewer.device
        gaussian_count = visual_data.selected_count
        total_points = int(visible_env_ids.size) * gaussian_count
        tiled_radii = np.tile(visual_data.radii, int(visible_env_ids.size))
        tiled_colors = np.tile(visual_data.colors, (int(visible_env_ids.size), 1))

        self._skinned_gaussian = _SkinnedGaussianRuntime(
            asset=asset,
            influence_indices=wp.array(visual_data.influence_indices, dtype=wp.int32, device=device),
            influence_weights=wp.array(visual_data.influence_weights, dtype=wp.float32, device=device),
            visible_env_ids=wp.array(visible_env_ids, dtype=wp.int32, device=device),
            radii=wp.array(tiled_radii, dtype=wp.float32, device=device),
            colors=wp.array(tiled_colors, dtype=wp.vec3f, device=device),
            points=wp.empty(total_points, dtype=wp.vec3f, device=device),
            gaussian_count=gaussian_count,
            total_points=total_points,
        )
        logger.info(
            "[SkinnedGaussianNewtonVisualizer] Loaded %d/%d Gaussian points per env (stride=%d), visible_envs=%d.",
            visual_data.selected_count,
            visual_data.source_count,
            visual_data.stride,
            visible_env_ids.size,
        )

    def _log_skinned_gaussians(self) -> None:
        from isaaclab_newton.physics import NewtonManager

        runtime = self._skinned_gaussian
        if runtime is None or self._viewer is None:
            return

        state = NewtonManager.get_state_0()
        particle_q = getattr(state, "particle_q", None) if state is not None else None
        if particle_q is None:
            return

        particle_offsets = getattr(runtime.asset.data, "_particle_offsets")
        wp.launch(
            skin_gaussian_points_kernel,
            dim=runtime.total_points,
            inputs=[
                particle_q,
                particle_offsets,
                runtime.visible_env_ids,
                runtime.influence_indices,
                runtime.influence_weights,
                runtime.gaussian_count,
            ],
            outputs=[runtime.points],
            device=self._viewer.device,
        )

        colors = runtime.colors if runtime.colors_pending_upload else None
        self._viewer.log_points(
            self.cfg.point_cloud_name,
            points=runtime.points,
            radii=runtime.radii,
            colors=colors,
            hidden=False,
        )
        runtime.colors_pending_upload = False
