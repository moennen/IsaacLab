# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local visualizers for Gaussian splats skinned to the deformable tet proxy."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg
from isaaclab_visualizers.newton_adapter import resolve_visible_env_indices

from isaaclab.utils.configclass import configclass
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

logger = logging.getLogger(__name__)

_ISAACLAB_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_SKINNED_GAUSSIAN_USD_PATH = os.environ.get(
    "ISAACLAB_DEXSUITE_SKINNED_GAUSSIAN_USD_PATH",
    str(_ISAACLAB_ROOT / "outputs" / "assets" / "blueHairRagdoll_skinned_gaussian_tet.usdc"),
)

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


@wp.kernel
def skin_gaussian_points_env_local_kernel(
    particle_q: wp.array(dtype=wp.vec3f),
    particle_offsets: wp.array(dtype=wp.int32),
    visible_env_ids: wp.array(dtype=wp.int32),
    env_position_offsets: wp.array(dtype=wp.vec3f),
    influence_indices: wp.array(dtype=wp.int32),
    influence_weights: wp.array(dtype=wp.float32),
    gaussian_count: int,
    out_points: wp.array(dtype=wp.vec3f),
):
    """Skin selected Gaussian centers into each authored Kit prim's local frame."""
    tid = wp.tid()
    env_slot = tid // gaussian_count
    gaussian_slot = tid - env_slot * gaussian_count
    env_id = visible_env_ids[env_slot]
    influence_offset = gaussian_slot * 4
    particle_offset = particle_offsets[env_id]

    i0 = particle_offset + influence_indices[influence_offset + 0]
    i1 = particle_offset + influence_indices[influence_offset + 1]
    i2 = particle_offset + influence_indices[influence_offset + 2]
    i3 = particle_offset + influence_indices[influence_offset + 3]

    w0 = influence_weights[influence_offset + 0]
    w1 = influence_weights[influence_offset + 1]
    w2 = influence_weights[influence_offset + 2]
    w3 = influence_weights[influence_offset + 3]

    world_point = particle_q[i0] * w0 + particle_q[i1] * w1 + particle_q[i2] * w2 + particle_q[i3] * w3
    out_points[tid] = world_point - env_position_offsets[env_id]


@wp.kernel
def skin_gaussian_points_to_fabric_kernel(
    fabric_positions: wp.fabricarrayarray(dtype=wp.vec3f),
    fabric_env_ids: wp.fabricarray(dtype=wp.uint32),
    fabric_asset_ids: wp.fabricarray(dtype=wp.uint32),
    particle_q: wp.array(dtype=wp.vec3f),
    particle_offsets: wp.array(dtype=wp.int32),
    env_position_offsets: wp.array(dtype=wp.vec3f),
    influence_indices: wp.array(dtype=wp.int32),
    influence_weights: wp.array(dtype=wp.float32),
    gaussian_counts: wp.array(dtype=wp.int32),
    max_gaussian_count: int,
):
    """Skin Gaussian fields directly into Fabric-owned USD positions on the GPU."""
    prim_slot = wp.tid()
    env_id = int(fabric_env_ids[prim_slot])
    asset_id = int(fabric_asset_ids[prim_slot])
    particle_offset = particle_offsets[env_id]
    gaussian_count = gaussian_counts[asset_id]
    asset_offset = asset_id * max_gaussian_count * 4
    for gaussian_slot in range(gaussian_count):
        influence_offset = asset_offset + gaussian_slot * 4
        i0 = particle_offset + influence_indices[influence_offset + 0]
        i1 = particle_offset + influence_indices[influence_offset + 1]
        i2 = particle_offset + influence_indices[influence_offset + 2]
        i3 = particle_offset + influence_indices[influence_offset + 3]
        world_point = (
            particle_q[i0] * influence_weights[influence_offset + 0]
            + particle_q[i1] * influence_weights[influence_offset + 1]
            + particle_q[i2] * influence_weights[influence_offset + 2]
            + particle_q[i3] * influence_weights[influence_offset + 3]
        )
        fabric_positions[prim_slot][gaussian_slot] = world_point - env_position_offsets[env_id]


@dataclass(frozen=True)
class SkinnedGaussianVisualData:
    """CPU-side Gaussian skinning data loaded from USD."""

    selected_indices: np.ndarray
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


@dataclass
class _SkinnedGaussianKitRuntime:
    """Runtime buffers and USD attrs owned by the skinned Gaussian Kit visualizer."""

    asset: object
    influence_indices: wp.array
    influence_weights: wp.array
    visible_env_ids: wp.array
    env_position_offsets: wp.array
    points: wp.array
    position_attrs: list[object]
    gaussian_count: int
    total_points: int


@dataclass
class _SkinnedGaussianKitFabricRuntime:
    """Fabric-backed GPU skinning state for all visible Kit Gaussian prims."""

    asset: object
    fabric_positions: object
    fabric_env_ids: object
    fabric_asset_ids: object
    env_position_offsets: wp.array
    influence_indices: wp.array
    influence_weights: wp.array
    gaussian_counts: wp.array
    max_gaussian_count: int
    prim_count: int


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


def _selected_sequence_values(value, selected: np.ndarray, source_count: int):
    try:
        value_count = len(value)
    except TypeError:
        return value

    if value_count == source_count:
        return [value[int(index)] for index in selected]
    if source_count > 0 and value_count > source_count and value_count % source_count == 0:
        stride = value_count // source_count
        return [value[int(index) * stride + offset] for index in selected for offset in range(stride)]
    return value


def _set_gaussian_casts_shadows(gaussian_prim) -> None:
    from pxr import Sdf, UsdGeom

    UsdGeom.PrimvarsAPI(gaussian_prim).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(False)


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
        selected_indices=np.ascontiguousarray(selected, dtype=np.int32),
        influence_indices=np.ascontiguousarray(influence_indices, dtype=np.int32),
        influence_weights=np.ascontiguousarray(influence_weights, dtype=np.float32),
        radii=np.ascontiguousarray(radii, dtype=np.float32),
        colors=np.ascontiguousarray(colors, dtype=np.float32),
        source_count=point_count,
        selected_count=int(selected.size),
        stride=stride,
    )


@configclass
class SkinnedGaussianKitVisualizerCfg(VisualizerCfg):
    """Kit overlay that renders skinned Gaussian splats for the deformable task."""

    visualizer_type: str = "kit"
    """Type identifier used so ``--visualizer kit`` selects this overlay."""

    skinned_gaussian_usd_path: str = DEFAULT_SKINNED_GAUSSIAN_USD_PATH
    """Combined Gaussian + tet USD containing ``newton:deformableSkin:*`` metadata."""

    skinned_gaussian_usd_paths: tuple[str, ...] = ()
    """Per-asset packaged Gaussian USDs indexed by ``newton:deformableAssetIndex``."""

    gaussian_prim_path: str | None = None
    """Optional Gaussian prim path. When omitted, the first ParticleField3DGaussianSplat prim is used."""

    deformable_asset_name: str = "deformable"
    """Scene asset name of the deformable object whose tet particles drive the Gaussians."""

    max_gaussians_per_env: int | None = 20_000
    """Maximum rendered Gaussian points per visible environment. Non-positive means all points."""

    hide_tet_visual_mesh: bool = True
    """Hide the coarse tetrahedral surface mesh when the Gaussian visual is available."""

    gaussian_scope_path: str = "SkinnedGaussianVisuals"
    """Per-environment scope path for task-authored Gaussian visualization prims."""

    gaussian_prim_name: str = "gaussians"
    """Per-environment Gaussian prim name under :attr:`gaussian_scope_path`."""

    def create_visualizer(self):
        """Create the task-specific Kit Gaussian overlay."""
        return SkinnedGaussianKitVisualizer(self)


class SkinnedGaussianKitVisualizer(BaseVisualizer):
    """Kit overlay that authors task-local Gaussian splats skinned to Newton particles."""

    cfg: SkinnedGaussianKitVisualizerCfg

    def __init__(self, cfg: SkinnedGaussianKitVisualizerCfg):
        super().__init__(cfg)
        self._skinned_gaussian_kits: dict[int, _SkinnedGaussianKitRuntime] = {}
        self._skinned_gaussian_kit_fabric: _SkinnedGaussianKitFabricRuntime | None = None
        self._kit_hidden_tet_mesh_paths: list[str] = []
        self._kit_gaussian_load_error: str | None = None
        self._env_ids: list[int] | None = None
        self._resolved_visible_env_ids: list[int] | None = None

    def initialize(self, scene_data_provider) -> None:
        """Initialize Kit visualizer and author task-local Gaussian prims."""
        scene_data_provider = self._set_scene_data_provider(scene_data_provider)
        self._env_ids = self._compute_visualized_env_ids()
        self._resolved_visible_env_ids = resolve_visible_env_indices(
            self._env_ids, self.cfg.max_visible_envs, scene_data_provider.num_envs
        )
        self._initialize_kit_skinned_gaussian_runtime()
        self._is_initialized = True

    def step(self, dt: float) -> None:
        """Update Gaussian positions before the standard Kit visualizer pumps the viewport."""
        if not self._is_initialized or self._is_closed:
            return
        self._update_kit_skinned_gaussians()

    def close(self) -> None:
        """Restore coarse mesh visibility before closing Kit resources."""
        self._restore_tet_visual_mesh_visibility()
        self._skinned_gaussian_kits.clear()
        self._skinned_gaussian_kit_fabric = None
        self._is_initialized = False
        self._is_closed = True

    def is_running(self) -> bool:
        """Return whether the overlay should continue updating."""
        return not self._is_closed

    def _initialize_kit_skinned_gaussian_runtime(self) -> None:
        from pxr import Sdf, Usd, UsdGeom, Vt

        if self._scene_data_provider is None:
            return
        stage = self._scene_data_provider.usd_stage
        if stage is None:
            return

        scene = self._scene_data_provider.get_interactive_scene()
        if scene is None:
            self._kit_gaussian_load_error = "interactive scene is unavailable"
            logger.warning("[SkinnedGaussianKitVisualizer] %s", self._kit_gaussian_load_error)
            return
        try:
            asset = scene[self.cfg.deformable_asset_name]
        except KeyError:
            self._kit_gaussian_load_error = f"scene has no deformable asset named '{self.cfg.deformable_asset_name}'"
            logger.warning("[SkinnedGaussianKitVisualizer] %s", self._kit_gaussian_load_error)
            return

        particle_offsets = getattr(asset.data, "_particle_offsets", None)
        particles_per_body = getattr(asset.data, "_particles_per_body", None)
        if particle_offsets is None or particles_per_body is None:
            self._kit_gaussian_load_error = "deformable asset does not expose Newton particle offsets"
            logger.warning("[SkinnedGaussianKitVisualizer] %s", self._kit_gaussian_load_error)
            return

        num_envs = self._scene_data_provider.num_envs
        env_ids = self._resolved_visible_env_ids
        visible_env_ids = (
            np.arange(num_envs, dtype=np.int32) if env_ids is None else np.asarray(env_ids, dtype=np.int32)
        )
        if visible_env_ids.size == 0:
            logger.info("[SkinnedGaussianKitVisualizer] No visible envs selected; Gaussian overlay disabled.")
            return

        asset_indices = getattr(asset, "_asset_indices", None)
        if asset_indices is None:
            self._kit_gaussian_load_error = "deformable asset has no per-environment asset indices"
            logger.warning("[SkinnedGaussianKitVisualizer] %s", self._kit_gaussian_load_error)
            return
        asset_indices = asset_indices.detach().cpu().numpy()
        usd_paths = self.cfg.skinned_gaussian_usd_paths or (self.cfg.skinned_gaussian_usd_path,)
        device = getattr(particle_offsets, "device", None) or "cuda:0"
        env_position_offsets = self._kit_env_position_offsets(scene, num_envs)
        visual_data_by_asset: dict[int, SkinnedGaussianVisualData] = {}
        for asset_index in np.unique(asset_indices[visible_env_ids]):
            if asset_index < 0 or asset_index >= len(usd_paths):
                logger.warning(
                    "[SkinnedGaussianKitVisualizer] No Gaussian USD configured for asset index %d.", asset_index
                )
                continue
            usd_path = Path(usd_paths[int(asset_index)]).expanduser()
            if not usd_path.is_file():
                logger.warning("[SkinnedGaussianKitVisualizer] Gaussian USD does not exist: '%s'.", usd_path)
                continue
            try:
                visual_data = load_skinned_gaussian_visual_data(
                    str(usd_path),
                    self.cfg.gaussian_prim_path,
                    max_gaussians_per_env=self.cfg.max_gaussians_per_env,
                    radius_scale=1.0,
                )
            except Exception as exc:
                logger.warning("[SkinnedGaussianKitVisualizer] Failed to load asset %d: %s", asset_index, exc)
                continue
            if int(visual_data.influence_indices.max(initial=0)) >= int(particles_per_body):
                raise ValueError(f"Skinning for asset {asset_index} exceeds the shared particle budget.")
            visual_data_by_asset[int(asset_index)] = visual_data
            source_stage = Usd.Stage.Open(str(usd_path))
            source_gaussian_prim = (
                source_stage.GetPrimAtPath(self.cfg.gaussian_prim_path)
                if self.cfg.gaussian_prim_path
                else _find_first_gaussian_prim(source_stage)
            )
            env_ids_for_asset = visible_env_ids[asset_indices[visible_env_ids] == asset_index]
            position_attrs = []
            zero_positions = Vt.Vec3fArray.FromNumpy(np.zeros((visual_data.selected_count, 3), dtype=np.float32))
            for env_id in env_ids_for_asset:
                env_scope_path = self._kit_gaussian_scope_path(int(env_id))
                prim_path = f"{env_scope_path}/{self.cfg.gaussian_prim_name}"
                stage.DefinePrim(env_scope_path, "Xform")
                gaussian_prim = stage.DefinePrim(prim_path, "ParticleField3DGaussianSplat")
                UsdGeom.Xformable(gaussian_prim).SetResetXformStack(True)
                gaussian_prim.CreateAttribute("newton:kitSkinEnvId", Sdf.ValueTypeNames.UInt, custom=True).Set(
                    int(env_id)
                )
                gaussian_prim.CreateAttribute("newton:kitSkinAssetId", Sdf.ValueTypeNames.UInt, custom=True).Set(
                    int(asset_index)
                )
                self._copy_kit_gaussian_attrs(source_gaussian_prim, gaussian_prim, visual_data)
                _set_gaussian_casts_shadows(gaussian_prim)
                position_attr = gaussian_prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray)
                position_attr.Set(zero_positions)
                position_attrs.append(position_attr)
            gaussian_count = visual_data.selected_count
            total_points = int(env_ids_for_asset.size) * gaussian_count
            self._skinned_gaussian_kits[int(asset_index)] = _SkinnedGaussianKitRuntime(
                asset=asset,
                influence_indices=wp.array(visual_data.influence_indices, dtype=wp.int32, device=device),
                influence_weights=wp.array(visual_data.influence_weights, dtype=wp.float32, device=device),
                visible_env_ids=wp.array(env_ids_for_asset, dtype=wp.int32, device=device),
                env_position_offsets=wp.array(env_position_offsets, dtype=wp.vec3f, device=device),
                points=wp.empty(total_points, dtype=wp.vec3f, device=device),
                position_attrs=position_attrs,
                gaussian_count=gaussian_count,
                total_points=total_points,
            )
        if self.cfg.hide_tet_visual_mesh:
            self._hide_tet_visual_mesh(stage, visible_env_ids)
        self._initialize_kit_fabric_runtime(
            asset=asset,
            particle_offsets=particle_offsets,
            env_position_offsets=env_position_offsets,
            visual_data_by_asset=visual_data_by_asset,
            num_assets=len(usd_paths),
        )
        self._update_kit_skinned_gaussians()

    def _kit_gaussian_scope_path(self, env_id: int) -> str:
        scope_path = str(self.cfg.gaussian_scope_path).strip()
        if not scope_path:
            scope_path = "SkinnedGaussianVisuals"
        if scope_path.startswith("/"):
            return f"{scope_path.rstrip('/')}/env_{env_id}"
        return f"/World/envs/env_{env_id}/{scope_path.strip('/')}"

    def _initialize_kit_fabric_runtime(
        self,
        *,
        asset,
        particle_offsets,
        env_position_offsets: np.ndarray,
        visual_data_by_asset: dict[int, SkinnedGaussianVisualData],
        num_assets: int,
    ) -> None:
        """Bind authored Gaussian position arrays to Fabric for GPU-only updates."""
        if not visual_data_by_asset:
            return
        try:
            from isaaclab_newton.physics import NewtonManager

            import usdrt

            fabric_stage = getattr(NewtonManager, "_usdrt_stage", None)
            if fabric_stage is None:
                return
            selection = fabric_stage.SelectPrims(
                require_attrs=[
                    (usdrt.Sdf.ValueTypeNames.Point3fArray, "positions", usdrt.Usd.Access.ReadWrite),
                    (usdrt.Sdf.ValueTypeNames.UInt, "newton:kitSkinEnvId", usdrt.Usd.Access.Read),
                    (usdrt.Sdf.ValueTypeNames.UInt, "newton:kitSkinAssetId", usdrt.Usd.Access.Read),
                ],
                device=str(getattr(particle_offsets, "device", "cuda:0")),
            )
            if selection.GetCount() == 0:
                return
            max_count = max(data.selected_count for data in visual_data_by_asset.values())
            indices = np.zeros((num_assets, max_count, 4), dtype=np.int32)
            weights = np.zeros((num_assets, max_count, 4), dtype=np.float32)
            counts = np.zeros(num_assets, dtype=np.int32)
            for asset_index, data in visual_data_by_asset.items():
                count = data.selected_count
                indices[asset_index, :count] = data.influence_indices.reshape(count, 4)
                weights[asset_index, :count] = data.influence_weights.reshape(count, 4)
                counts[asset_index] = count
            device = getattr(particle_offsets, "device", None) or "cuda:0"
            self._skinned_gaussian_kit_fabric = _SkinnedGaussianKitFabricRuntime(
                asset=asset,
                fabric_positions=wp.fabricarrayarray(data=selection, attrib="positions", dtype=wp.vec3f),
                fabric_env_ids=wp.fabricarray(data=selection, attrib="newton:kitSkinEnvId", dtype=wp.uint32),
                fabric_asset_ids=wp.fabricarray(data=selection, attrib="newton:kitSkinAssetId", dtype=wp.uint32),
                env_position_offsets=wp.array(env_position_offsets, dtype=wp.vec3f, device=device),
                influence_indices=wp.array(indices.reshape(-1), dtype=wp.int32, device=device),
                influence_weights=wp.array(weights.reshape(-1), dtype=wp.float32, device=device),
                gaussian_counts=wp.array(counts, dtype=wp.int32, device=device),
                max_gaussian_count=max_count,
                prim_count=selection.GetCount(),
            )
            # The Fabric selection is a subset of the authored Gaussian fields;
            # only use it when every visible env field was bound successfully.
            if selection.GetCount() != sum(
                len(runtime.position_attrs) for runtime in self._skinned_gaussian_kits.values()
            ):
                logger.warning("[SkinnedGaussianKitVisualizer] Fabric selection is incomplete; using USD fallback.")
                self._skinned_gaussian_kit_fabric = None
            else:
                logger.info(
                    "[SkinnedGaussianKitVisualizer] Using GPU Fabric skinning for %d Gaussian fields.",
                    selection.GetCount(),
                )
        except Exception as exc:
            logger.info("[SkinnedGaussianKitVisualizer] Fabric skinning unavailable; using USD fallback: %s", exc)
            self._skinned_gaussian_kit_fabric = None

    def _kit_env_position_offsets(self, scene, num_envs: int) -> np.ndarray:
        env_position_offsets = np.zeros((num_envs, 3), dtype=np.float32)
        scope_path = str(self.cfg.gaussian_scope_path).strip()
        if scope_path.startswith("/"):
            return env_position_offsets

        env_origins = getattr(scene, "env_origins", None)
        if env_origins is None:
            return env_position_offsets
        if hasattr(env_origins, "detach"):
            env_origins = env_origins.detach().cpu().numpy()
        env_origins = np.asarray(env_origins, dtype=np.float32)
        if env_origins.ndim != 2 or env_origins.shape[1] != 3:
            logger.warning(
                "[SkinnedGaussianKitVisualizer] Ignoring unexpected env_origins shape: %s",
                env_origins.shape,
            )
            return env_position_offsets
        env_position_offsets[: min(num_envs, env_origins.shape[0])] = env_origins[:num_envs]
        return env_position_offsets

    def _copy_kit_gaussian_attrs(self, source_prim, gaussian_prim, visual_data: SkinnedGaussianVisualData) -> None:
        for source_attr in source_prim.GetAttributes():
            name = source_attr.GetName()
            if name == "positions" or name.startswith("newton:deformableSkin:"):
                continue
            value = source_attr.Get()
            if value is None:
                continue
            value = _selected_sequence_values(value, visual_data.selected_indices, visual_data.source_count)
            dst_attr = gaussian_prim.CreateAttribute(name, source_attr.GetTypeName(), custom=source_attr.IsCustom())
            dst_attr.Set(value)

    def _hide_tet_visual_mesh(self, stage, visible_env_ids: np.ndarray) -> None:
        from pxr import UsdGeom

        for env_id in visible_env_ids:
            mesh_path = f"/World/envs/env_{int(env_id)}/Deformable/geometry/visual_mesh"
            prim = stage.GetPrimAtPath(mesh_path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
                self._kit_hidden_tet_mesh_paths.append(mesh_path)

    def _restore_tet_visual_mesh_visibility(self) -> None:
        if not self._kit_hidden_tet_mesh_paths or self._scene_data_provider is None:
            return
        from pxr import UsdGeom

        stage = self._scene_data_provider.usd_stage
        if stage is None:
            return
        for mesh_path in self._kit_hidden_tet_mesh_paths:
            prim = stage.GetPrimAtPath(mesh_path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeVisible()
        self._kit_hidden_tet_mesh_paths.clear()

    def _update_kit_skinned_gaussians(self) -> None:
        if not self._skinned_gaussian_kits:
            return

        from isaaclab_newton.physics import NewtonManager

        from pxr import Vt

        state = NewtonManager.get_state_0()
        particle_q = getattr(state, "particle_q", None) if state is not None else None
        if particle_q is None:
            return

        fabric_runtime = self._skinned_gaussian_kit_fabric
        if fabric_runtime is not None:
            particle_offsets = getattr(fabric_runtime.asset.data, "_particle_offsets")
            wp.launch(
                skin_gaussian_points_to_fabric_kernel,
                dim=fabric_runtime.prim_count,
                inputs=[
                    fabric_runtime.fabric_positions,
                    fabric_runtime.fabric_env_ids,
                    fabric_runtime.fabric_asset_ids,
                    particle_q,
                    particle_offsets,
                    fabric_runtime.env_position_offsets,
                    fabric_runtime.influence_indices,
                    fabric_runtime.influence_weights,
                    fabric_runtime.gaussian_counts,
                    fabric_runtime.max_gaussian_count,
                ],
                device=fabric_runtime.influence_indices.device,
            )
            return

        for runtime in self._skinned_gaussian_kits.values():
            particle_offsets = getattr(runtime.asset.data, "_particle_offsets")
            wp.launch(
                skin_gaussian_points_env_local_kernel,
                dim=runtime.total_points,
                inputs=[
                    particle_q,
                    particle_offsets,
                    runtime.visible_env_ids,
                    runtime.env_position_offsets,
                    runtime.influence_indices,
                    runtime.influence_weights,
                    runtime.gaussian_count,
                ],
                outputs=[runtime.points],
                device=runtime.points.device,
            )
            points = runtime.points.numpy().reshape(len(runtime.position_attrs), runtime.gaussian_count, 3)
            for env_points, position_attr in zip(points, runtime.position_attrs):
                position_attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(env_points, dtype=np.float32)))


@configclass
class SkinnedGaussianNewtonVisualizerCfg(NewtonVisualizerCfg):
    """Newton visualizer that overlays skinned Gaussian positions for the deformable task."""

    skinned_gaussian_usd_path: str = DEFAULT_SKINNED_GAUSSIAN_USD_PATH
    """Combined Gaussian + tet USD containing ``newton:deformableSkin:*`` metadata."""

    skinned_gaussian_usd_paths: tuple[str, ...] = ()
    """Per-asset packaged Gaussian USDs indexed by ``newton:deformableAssetIndex``."""

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
        self._skinned_gaussians: dict[int, _SkinnedGaussianRuntime] = {}
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

        num_envs = self._scene_data_provider.num_envs
        env_ids = self._resolved_visible_env_ids
        visible_env_ids = (
            np.arange(num_envs, dtype=np.int32) if env_ids is None else np.asarray(env_ids, dtype=np.int32)
        )
        if visible_env_ids.size == 0:
            logger.info("[SkinnedGaussianNewtonVisualizer] No visible envs selected; Gaussian overlay disabled.")
            return

        asset_indices = getattr(asset, "_asset_indices", None)
        if asset_indices is None:
            logger.warning("[SkinnedGaussianNewtonVisualizer] Deformable asset has no per-environment asset indices.")
            return
        asset_indices = asset_indices.detach().cpu().numpy()
        usd_paths = self.cfg.skinned_gaussian_usd_paths or (self.cfg.skinned_gaussian_usd_path,)
        visible_asset_indices = asset_indices[visible_env_ids]
        logger.info(
            "[SkinnedGaussianNewtonVisualizer] visible env asset IDs: %s; configured Gaussian USDs: %d.",
            np.unique(visible_asset_indices, return_counts=True),
            len(usd_paths),
        )
        for asset_index in np.unique(visible_asset_indices):
            if asset_index < 0 or asset_index >= len(usd_paths):
                logger.warning(
                    "[SkinnedGaussianNewtonVisualizer] No Gaussian USD configured for asset index %d.", asset_index
                )
                continue
            usd_path = Path(usd_paths[int(asset_index)]).expanduser()
            if not usd_path.is_file():
                logger.warning("[SkinnedGaussianNewtonVisualizer] Gaussian USD does not exist: '%s'.", usd_path)
                continue
            visual_data = load_skinned_gaussian_visual_data(
                str(usd_path),
                self.cfg.gaussian_prim_path,
                max_gaussians_per_env=self.cfg.max_gaussians_per_env,
                radius_scale=self.cfg.radius_scale,
                min_radius=self.cfg.min_radius,
            )
            if int(visual_data.influence_indices.max(initial=0)) >= int(particles_per_body):
                raise ValueError(f"Skinning for asset {asset_index} exceeds the shared particle budget.")
            env_ids = visible_env_ids[asset_indices[visible_env_ids] == asset_index]
            device = self._viewer.device
            gaussian_count = visual_data.selected_count
            total_points = int(env_ids.size) * gaussian_count
            self._skinned_gaussians[int(asset_index)] = _SkinnedGaussianRuntime(
                asset=asset,
                influence_indices=wp.array(visual_data.influence_indices, dtype=wp.int32, device=device),
                influence_weights=wp.array(visual_data.influence_weights, dtype=wp.float32, device=device),
                visible_env_ids=wp.array(env_ids, dtype=wp.int32, device=device),
                radii=wp.array(np.tile(visual_data.radii, int(env_ids.size)), dtype=wp.float32, device=device),
                colors=wp.array(np.tile(visual_data.colors, (int(env_ids.size), 1)), dtype=wp.vec3f, device=device),
                points=wp.empty(total_points, dtype=wp.vec3f, device=device),
                gaussian_count=gaussian_count,
                total_points=total_points,
            )
            logger.info(
                "[SkinnedGaussianNewtonVisualizer] loaded asset %d: %d Gaussian points across %d visible envs.",
                asset_index,
                gaussian_count,
                env_ids.size,
            )

    def _log_skinned_gaussians(self) -> None:
        from isaaclab_newton.physics import NewtonManager

        if not self._skinned_gaussians or self._viewer is None:
            return

        state = NewtonManager.get_state_0()
        particle_q = getattr(state, "particle_q", None) if state is not None else None
        if particle_q is None:
            return

        for asset_index, runtime in self._skinned_gaussians.items():
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
                f"{self.cfg.point_cloud_name}/{asset_index}",
                points=runtime.points,
                radii=runtime.radii,
                colors=colors,
                hidden=False,
            )
            runtime.colors_pending_upload = False
