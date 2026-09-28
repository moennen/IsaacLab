# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gaussian-twin asset facade for Kaolin RKPM Simplicits simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.assets.asset_base import AssetBase
from isaaclab.assets.deformable_object.deformable_object_cfg import DeformableObjectCfg
from isaaclab.sim import find_matching_prims
from isaaclab.utils.configclass import configclass

from .gaussian_twin import gaussian_twin_updates_enabled


@wp.kernel
def _write_gaussian_positions(
    positions: wp.array(dtype=wp.vec3),
    transforms: wp.array(dtype=wp.transformf),
):
    """Overwrite Gaussian centers while preserving their bind-pose rotations."""
    index = wp.tid()
    transform = transforms[index]
    transforms[index] = wp.transformf(positions[index], wp.transform_get_rotation(transform))


@configclass
class GaussianTwinSimplicitsCfg(DeformableObjectCfg):
    """Spawn configuration for one RKPM-packaged Gaussian-twin slot.

    The object does not use Isaac Lab's TetMesh state buffers, but it must be
    classified as a deformable scene asset.  That is the scene category whose
    ``reset()`` and ``update()`` hooks are invoked each simulation step.  In
    particular, ``update()`` streams the current RKPM rendered positions to
    the Gaussian field.
    """

    class_type: type = "{DIR}.gaussian_twin_simplicits:GaussianTwinSimplicitsObject"
    slot_index: int = 0
    num_objects: int = 1
    num_slots: int = 1
    splat_deformation: str = "position"
    """Supported Gaussian deformation stream; RKPM currently provides positions only."""


@dataclass
class _GaussianBinding:
    object_id: int
    transforms: wp.array
    opacities: wp.array
    rest_opacities: wp.array
    hidden_opacities: wp.array


class GaussianTwinSimplicitsObject(AssetBase):
    """Synchronize one Gaussian field with Kaolin's rendered RKPM points."""

    _selection_by_model_env: dict[tuple[int, int, int, int], np.ndarray] = {}

    def __init__(self, cfg: GaussianTwinSimplicitsCfg):
        if cfg.splat_deformation != "position":
            raise ValueError(
                "GaussianTwinSimplicitsObject supports position-only Gaussian streaming; "
                f"got '{cfg.splat_deformation}'."
            )
        self._slot_index = int(cfg.slot_index)
        self._num_objects = int(cfg.num_objects)
        self._num_slots = int(cfg.num_slots)
        self._bindings: list[_GaussianBinding] = []
        self._num_instances = 0
        super().__init__(cfg)

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def data(self):
        return None

    def _initialize_impl(self):
        import newton

        from isaaclab_contrib.custom_coupling.coupled_mjwarp_simplicits_manager import (
            NewtonCoupledMJWarpSimplicitsManager,
        )

        model = NewtonManager.get_model()
        if model is None:
            raise RuntimeError("Newton model is unavailable while initializing the Simplicits Gaussian twin.")
        _scene, object_ids = NewtonCoupledMJWarpSimplicitsManager.get_simplicits_scene_and_object_ids()
        self._num_instances = len(object_ids)
        labels = list(getattr(model, "shape_label", ()))
        matching = [
            (shape_index, source)
            for shape_index, source in enumerate(model.shape_source)
            if isinstance(source, newton.Gaussian) and f"/GaussianTwin_{self._slot_index}/" in str(labels[shape_index])
        ]
        if len(matching) != self._num_instances:
            raise RuntimeError(
                f"Expected one Simplicits Gaussian field per env for slot {self._slot_index}; "
                f"found {len(matching)} for {self._num_instances} environments."
            )
        shape_transforms = model.shape_transform.numpy()
        shape_scales = model.shape_scale.numpy()
        self._bindings.clear()
        for env_id, (shape_index, source) in enumerate(matching):
            source_data = source.warp_data
            if source_data is None:
                raise RuntimeError("Newton did not finalize the Gaussian shape for the Simplicits visual field.")
            source._newton_dynamic_attributes = ("positions",)
            self._bindings.append(
                _GaussianBinding(
                    object_id=object_ids[env_id][self._slot_index],
                    transforms=source_data.transforms,
                    opacities=source_data.opacities,
                    rest_opacities=wp.clone(source_data.opacities),
                    hidden_opacities=wp.zeros_like(source_data.opacities),
                )
            )
            # Kaolin gives world-space rendered positions.  Avoid applying the
            # cloned environment/source transforms a second time in RTX.
            shape_transforms[shape_index] = np.asarray(wp.transform_identity(), dtype=np.float32)
            shape_scales[shape_index] = (1.0, 1.0, 1.0)
        wp.copy(model.shape_transform, wp.array(shape_transforms, dtype=wp.transform, device=self.device))
        wp.copy(model.shape_scale, wp.array(shape_scales, dtype=wp.vec3f, device=self.device))
        self._set_proxy_mesh_visibility()

    def _set_proxy_mesh_visibility(self) -> None:
        from pxr import Usd, UsdGeom

        for prim in find_matching_prims(self.cfg.prim_path):
            for child in Usd.PrimRange(prim):
                if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.TetMesh):
                    UsdGeom.Imageable(child).MakeInvisible()

    def _selection(self, env_id: int, *, refresh: bool) -> np.ndarray:
        key = (id(NewtonManager.get_model()), self._num_slots, self._num_objects, env_id)
        selected = self._selection_by_model_env.get(key)
        if refresh or selected is None:
            selected = np.random.choice(self._num_slots, size=self._num_objects, replace=False)
            self._selection_by_model_env[key] = selected
        return selected

    def reset(self, env_ids=None, env_mask=None) -> None:
        from isaaclab_contrib.custom_coupling.coupled_mjwarp_simplicits_manager import (
            NewtonCoupledMJWarpSimplicitsManager,
        )

        if env_ids is None:
            if env_mask is None:
                env_ids = range(self._num_instances)
            else:
                mask = torch.as_tensor(env_mask, device="cpu", dtype=torch.bool).reshape(-1)
                if mask.numel() != self._num_instances:
                    raise ValueError(f"env_mask must have {self._num_instances} entries, got {mask.numel()}.")
                env_ids = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        elif env_mask is not None:
            mask = torch.as_tensor(env_mask, device="cpu", dtype=torch.bool).reshape(-1)
            env_ids = [int(value) for value in env_ids if bool(mask[int(value)])]
        ids = [int(value) for value in torch.as_tensor(list(env_ids), device="cpu").reshape(-1).tolist()]
        if not ids:
            return
        active_by_env = [self._slot_index in self._selection(env_id, refresh=self._slot_index == 0) for env_id in ids]
        # The manager applies every selected environment to one cloned reduced
        # state and uploads it once.  Calling it per environment used to clone
        # and rewrite the entire scene for every reset ID.
        NewtonCoupledMJWarpSimplicitsManager.reset_gaussian_twin_slot(
            self._slot_index,
            torch.as_tensor(ids, device=self.device, dtype=torch.long),
            torch.as_tensor(active_by_env, device=self.device, dtype=torch.bool),
        )
        for env_id, active in zip(ids, active_by_env, strict=True):
            binding = self._bindings[env_id]
            wp.copy(binding.opacities, binding.rest_opacities if active else binding.hidden_opacities)

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        del dt
        if not gaussian_twin_updates_enabled() or not self._bindings:
            return
        from isaaclab_contrib.custom_coupling.coupled_mjwarp_simplicits_manager import (
            NewtonCoupledMJWarpSimplicitsManager,
        )

        scene, _object_ids = NewtonCoupledMJWarpSimplicitsManager.get_simplicits_scene_and_object_ids()
        for binding in self._bindings:
            points = scene.get_object_deformed_pts(binding.object_id, points="rendered")
            if points.shape[0] != len(binding.transforms):
                raise RuntimeError("Kaolin rendered-point count changed after packaging.")
            # The old Torch ``cat`` materialized a seven-float transform for
            # every Gaussian every frame (over 150k for the toy assets). Keep
            # rotations in the GPU-resident Newton buffer and update only the
            # three dynamic coordinates through one Warp launch.
            wp.launch(
                _write_gaussian_positions,
                dim=len(binding.transforms),
                inputs=[wp.from_torch(points.contiguous(), dtype=wp.vec3), binding.transforms],
                device=self.device,
            )
