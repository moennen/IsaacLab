# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-owned Fabric resource shared by Kit and Isaac RTX."""

from __future__ import annotations

from dataclasses import field
from typing import Any

import numpy as np
import warp as wp

import usdrt
import usdrt.hierarchy
from pxr import Sdf, Usd, UsdGeom, UsdUtils, Vt

from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
from isaaclab.scene_data.deformable_gaussian_splats import read_gaussian_splats_attribute
from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import TimestampedBuffer


@wp.kernel(enable_backward=False)
def _capture_scales(
    matrices: wp.fabricarray(dtype=wp.mat44d),
    indices: wp.fabricarray(dtype=wp.int32),
    scales: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    matrix = wp.mat44f(matrices[i])
    scales[indices[i]] = wp.vec3f(
        wp.length(wp.vec3f(matrix[0, 0], matrix[0, 1], matrix[0, 2])),
        wp.length(wp.vec3f(matrix[1, 0], matrix[1, 1], matrix[1, 2])),
        wp.length(wp.vec3f(matrix[2, 0], matrix[2, 1], matrix[2, 2])),
    )


@wp.kernel(enable_backward=False)
def _splat_destination_rows(tags: Any, rows: wp.array(dtype=wp.int32)):
    row = wp.tid()
    rows[tags[row]] = row


@wp.kernel(enable_backward=False)
def _write_fabric_splats(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    scales: wp.array(dtype=wp.vec3f),
    coefficients: wp.array(dtype=wp.vec3f),
    count: int,
    basis_count: int,
    rows: wp.array(dtype=wp.int32),
    out_positions: Any,
    out_orientations: Any,
    out_scales: Any,
    out_coefficients: Any,
):
    splat = wp.tid()
    row, index = rows[splat // count], splat % count
    out_positions[row, index] = positions[splat]
    out_orientations[row, index] = orientations[splat]
    out_scales[row, index] = scales[splat]
    for basis in range(basis_count):
        out_coefficients[row, index * basis_count + basis] = coefficients[
            (splat * basis_count + basis) % coefficients.shape[0]
        ]


class FabricBackend:
    """Own one stage's native Fabric handles and shared transform/geometry bindings.

    Consumers supply the simulation's SDP explicitly. Its producer layout stays fixed for the
    binding's lifetime; it is not part of the native stage/device identity.
    """

    def __init__(self, cfg: FabricBackendCfg):
        self.device = cfg.device
        self.usd_stage = cfg.stage
        self.stage = usdrt.Usd.Stage.Attach(UsdUtils.StageCache.Get().GetId(cfg.stage).ToLongInt())
        self.hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            self.stage.GetFabricId(), self.stage.GetStageIdAsStageId()
        )
        self.transforms: SceneDataFormat.FabricMatrix44 | None = None
        self._selection = self._write_selection = None
        self._mapping = self._scales = None
        self._transforms_timestamp = -1
        self._geometry_bindings = None
        self._gaussian_splats_timestamp = -1
        self._gaussian_splats_paths: set[str] = set()
        self._gaussian_splats_bindings: dict[tuple[str, ...], tuple] = {}

    def bind_transforms(self, provider: SceneDataProvider) -> None:
        """Bind the initialized simulation's rigid destinations once; native Fabric needs no conversion binding."""
        if self.transforms is not None:
            return
        if SceneDataFormat.FabricMatrix44 in provider.backend.native_transform_formats:
            self.transforms = SceneDataFormat.FabricMatrix44()
            return

        stage = self.stage
        stage.SynchronizeToFabric()
        for path in provider.backend.transform_paths:
            prim = stage.GetPrimAtPath(path)
            if prim:
                usdrt.Rt.Xformable(prim).SetWorldXformFromUsd()
            else:
                prim = stage.DefinePrim(path, "Xform")
                usdrt.Rt.Xformable(prim).CreateFabricHierarchyWorldMatrixAttr()
            prim.AddAppliedSchema("PhysicsRigidBodyAPI")
        self.hierarchy.update_world_xforms()
        for index, path in enumerate(provider.backend.transform_paths):
            prim = stage.GetPrimAtPath(path)
            prim.CreateAttribute("isaaclab:transformIndex", usdrt.Sdf.ValueTypeNames.Int, custom=True).Set(index)
            # Physics publishes absolute body poses; only visual descendants inherit them.
            self.hierarchy.set_reset_xform_stack(prim.GetPath().fabricPath, True)
        attrs = [
            (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read),
            (usdrt.Sdf.ValueTypeNames.Int, "isaaclab:transformIndex", usdrt.Usd.Access.Read),
            (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:localMatrix", usdrt.Usd.Access.Read),
        ]
        self._selection = stage.SelectPrims(require_attrs=attrs, device=self.device)
        self._write_selection = stage.SelectPrims(
            require_attrs=[*attrs[:-1], (*attrs[-1][:2], usdrt.Usd.Access.ReadWrite)], device=self.device
        )
        self._scales = wp.empty(provider.transform_count, dtype=wp.vec3f, device=self.device)
        self.transforms = SceneDataFormat.FabricMatrix44()

    def update_transforms(self, provider: SceneDataProvider) -> None:
        """Request SDP transforms and propagate converted body matrices to visual descendants."""
        self.bind_transforms(provider)
        changed = self._selection is not None and self._selection.PrepareForReuse()
        if self._selection is not None and (changed or self.transforms.matrices is None):
            self._write_selection.PrepareForReuse()
            self._mapping = wp.fabricarray(self._selection, "isaaclab:transformIndex")
            if self.transforms.matrices is None:
                wp.launch(
                    _capture_scales,
                    dim=len(self._mapping),
                    inputs=[wp.fabricarray(self._selection, "omni:fabric:worldMatrix"), self._mapping],
                    outputs=[self._scales],
                    device=self._scales.device,
                )
            self.transforms = SceneDataFormat.FabricMatrix44()
            self.transforms.matrices = wp.fabricarray(self._write_selection, "omni:fabric:localMatrix")
        provider.get_transforms(self.transforms, self._mapping, scales=self._scales)
        timestamp = provider.backend.transforms_timestamp
        if self._selection is not None and (changed or self._transforms_timestamp != timestamp):
            self._write_selection.PrepareForReuse()
            device = self._scales.device
            wp.synchronize_stream(device)
            if not self.hierarchy.update_world_xforms_gpu(not changed and self._transforms_timestamp != -1):
                raise RuntimeError("Fabric GPU transform hierarchy update failed.")
            wp.synchronize_device(device)
        self._transforms_timestamp = timestamp

    def update_geometries(self, provider: SceneDataProvider, frame: int) -> None:
        """Request world-space visual vertices [m] directly into due Fabric destinations."""
        # The native transform request refreshes the whole Fabric stage, including geometry.
        if SceneDataFormat.FabricMatrix44 in provider.backend.native_transform_formats:
            return
        batches = provider.backend.get_geometry_batches()
        if self._geometry_bindings is None:
            self._bind_geometries(provider, tuple(path for _, ranges in batches for path in ranges))
        timestamp = provider.backend.geometry_timestamp
        for index, binding in enumerate(self._geometry_bindings):
            selections, frequency, cached, offsets, frame_last_update = binding
            selection, write_selection = selections
            changed = selection.PrepareForReuse()
            if (
                cached.data is not None
                and not changed
                and (timestamp == cached.timestamp or frequency > 1 and frame - frame_last_update < frequency)
            ):
                continue
            write_selection.PrepareForReuse()
            output = cached.data
            if output is None or changed:
                cached.data = None  # Retry the binding too if conversion fails after the selection changes.
                indices = wp.fabricarray(selection, f"isaaclab:geometryIndex:group{index}").numpy()
                rows = {int(value): row for row, value in enumerate(indices)}
                offsets = {path: rows[row] for row, path in enumerate(offsets)}
                output = SceneDataFormat.FabricPoints()
                output.points = wp.fabricarrayarray(data=write_selection, attrib="points", dtype=wp.vec3f)
            provider.get_geometry_points(output=output, offsets=offsets)
            cached.data, cached.timestamp = output, provider.backend.geometry_timestamp
            self._geometry_bindings[index] = (selections, frequency, cached, offsets, frame)

    def update_gaussian_splats(self, provider: SceneDataProvider) -> None:
        """Write CUDA Gaussian arrays to Fabric, or CPU arrays through USD."""
        states = provider.get_deformable_gaussian_splats()
        timestamp = provider.backend.geometry_timestamp
        if not states or self._gaussian_splats_timestamp == timestamp:
            return
        device_states = all(state.positions.device.is_cuda for state in states.values())
        if device_states != any(state.positions.device.is_cuda for state in states.values()):
            raise ValueError("Gaussian states for one Fabric update must share a device representation.")
        self.prepare_gaussian_splats(tuple(states), synchronize=device_states)
        if device_states:
            for paths, state, _ in states.batches:
                binding = self._gaussian_splats_bindings.get(paths)
                new_binding = binding is None
                if binding is None:
                    tag = f"isaaclab:splatIndex:group{len(self._gaussian_splats_bindings)}"
                    for index, path in enumerate(paths):
                        self.stage.GetPrimAtPath(path).CreateAttribute(
                            tag, usdrt.Sdf.ValueTypeNames.Int, custom=True
                        ).Set(index)
                    attributes = [
                        (usdrt.Sdf.ValueTypeNames.Int, tag, usdrt.Usd.Access.Read),
                        (usdrt.Sdf.ValueTypeNames.Point3fArray, "positions", usdrt.Usd.Access.ReadWrite),
                        (usdrt.Sdf.ValueTypeNames.QuatfArray, "orientations", usdrt.Usd.Access.ReadWrite),
                        (usdrt.Sdf.ValueTypeNames.Float3Array, "scales", usdrt.Usd.Access.ReadWrite),
                    ]
                    if state.sh_coefficients is not None:
                        attributes.append(
                            (
                                usdrt.Sdf.ValueTypeNames.Float3Array,
                                "radiance:sphericalHarmonicsCoefficients",
                                usdrt.Usd.Access.ReadWrite,
                            )
                        )
                    selection = self.stage.SelectPrims(require_attrs=attributes, device=self.device)
                    rows = wp.empty(len(paths), dtype=wp.int32, device=self.device)
                    binding = (selection, rows, tag)
                    self._gaussian_splats_bindings[paths] = binding
                selection, rows, tag = binding
                changed = selection.PrepareForReuse()
                if new_binding or changed:
                    wp.launch(
                        _splat_destination_rows,
                        dim=len(paths),
                        inputs=[wp.fabricarray(selection, tag, dtype=wp.int32), rows],
                        device=self.device,
                    )
                positions = wp.fabricarrayarray(selection, "positions", dtype=wp.vec3f)
                orientations = wp.fabricarrayarray(selection, "orientations", dtype=wp.quatf)
                scales = wp.fabricarrayarray(selection, "scales", dtype=wp.vec3f)
                coefficients = (
                    wp.fabricarrayarray(selection, "radiance:sphericalHarmonicsCoefficients", dtype=wp.vec3f)
                    if state.sh_coefficients is not None
                    else positions
                )
                count = len(state.positions) // len(paths)
                basis_count = (
                    len(states[paths[0]].sh_coefficients) // count if state.sh_coefficients is not None and count else 0
                )
                wp.launch(
                    _write_fabric_splats,
                    dim=len(state.positions),
                    inputs=[
                        state.positions,
                        state.orientations,
                        state.scales,
                        state.sh_coefficients if state.sh_coefficients is not None else state.positions,
                        count,
                        basis_count,
                        rows,
                        positions,
                        orientations,
                        scales,
                        coefficients,
                    ],
                    device=self.device,
                )
            wp.synchronize_stream(self.device)
        else:
            for path, state in states.items():
                prim = self.usd_stage.GetPrimAtPath(path)
                prim.GetAttribute("positions").Set(Vt.Vec3fArray.FromNumpy(state.positions.numpy()))
                prim.GetAttribute("orientations").Set(Vt.QuatfArray.FromNumpy(state.orientations.numpy()))
                prim.GetAttribute("scales").Set(Vt.Vec3fArray.FromNumpy(state.scales.numpy()))
                if state.sh_coefficients is not None:
                    prim.GetAttribute("radiance:sphericalHarmonicsCoefficients").Set(
                        Vt.Vec3fArray.FromNumpy(state.sh_coefficients.numpy())
                    )
            self.stage.SynchronizeToFabric()
        self._gaussian_splats_timestamp = timestamp

    def prepare_gaussian_splats(self, paths: tuple[str, ...], *, synchronize: bool = True) -> None:
        """Reset Gaussian transforms and populate Fabric before dynamic device writes."""
        new_paths = [path for path in paths if path not in self._gaussian_splats_paths]
        for path in new_paths:
            prim = self.usd_stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"Deformable Gaussian prim {path!r} is absent from the RTX stage.")
            positions = read_gaussian_splats_attribute(prim, "positions")
            count = len(positions) if positions is not None else 0
            for name, dtype, default in (
                ("positions", Sdf.ValueTypeNames.Point3fArray, np.zeros((count, 3), dtype=np.float32)),
                ("orientations", Sdf.ValueTypeNames.QuatfArray, np.tile([0, 0, 0, 1], (count, 1))),
                ("scales", Sdf.ValueTypeNames.Float3Array, np.ones((count, 3), dtype=np.float32)),
                ("radiance:sphericalHarmonicsCoefficients", Sdf.ValueTypeNames.Float3Array, None),
            ):
                if prim.GetAttribute(name).Get() is not None:
                    continue
                value = read_gaussian_splats_attribute(prim, name)
                if value is not None and name == "orientations":
                    value = [[*q.GetImaginary(), q.GetReal()] for q in value]
                value = default if value is None else np.asarray(value, dtype=np.float32)
                if value is not None:
                    array_type = Vt.QuatfArray if name == "orientations" else Vt.Vec3fArray
                    prim.CreateAttribute(name, dtype).Set(array_type.FromNumpy(np.asarray(value, dtype=np.float32)))
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            xformable.SetResetXformStack(True)
        if new_paths and synchronize:
            self.stage.SynchronizeToFabric()
        self._gaussian_splats_paths.update(new_paths)

    def _bind_geometries(self, provider: SceneDataProvider, paths: tuple[str, ...]) -> None:
        """Bind declared destinations by device and cadence; SDP owns all data movement."""
        # Foreign physics publishes world points. Author the sink once so Kit's USD refresh agrees.
        groups = {}
        for path in paths:
            geometry = UsdGeom.Xformable(self.usd_stage.GetPrimAtPath(path))
            geometry.ClearXformOpOrder()
            geometry.SetResetXformStack(True)
            prim_type = geometry.GetPrim().GetTypeName()
            if prim_type == "Mesh":
                device, frequency = self.device, 1
            elif prim_type in {"BasisCurves", "Points"}:
                # RTX Hydra reads these points from CPU Fabric (BasisCurves: NVBug 6502662).
                device = "cpu"
                frequency = geometry.GetPrim().GetAttribute("isaaclab:pointsUpdateFrequency").Get() or 1
            else:
                raise TypeError(f"Unsupported Fabric point destination {path}: {prim_type}.")
            groups.setdefault((device, frequency), []).append(path)
        self.stage.SynchronizeToFabric()
        self._geometry_bindings = []
        for index, (group, group_paths) in enumerate(groups.items()):
            device, frequency = group
            tag = f"isaaclab:geometryIndex:group{index}"
            for row, path in enumerate(group_paths):
                prim = self.stage.GetPrimAtPath(path)
                usdrt.Rt.Xformable(prim).SetWorldXformFromUsd()
                prim.CreateAttribute(tag, usdrt.Sdf.ValueTypeNames.Int, custom=True).Set(row)
            attrs = [
                (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Int, tag, usdrt.Usd.Access.Read),
            ]
            selection = self.stage.SelectPrims(require_attrs=attrs, device=device)
            write_selection = self.stage.SelectPrims(
                require_attrs=[(*attrs[0][:2], usdrt.Usd.Access.ReadWrite), attrs[1]], device=device
            )
            offsets = dict.fromkeys(group_paths, 0)
            self._geometry_bindings.append(
                ((selection, write_selection), frequency, TimestampedBuffer(), offsets, -frequency)
            )

    def close(self) -> None:
        """Release stage-bound selections and borrowed SDP buffers."""
        self.transforms = self._selection = self._write_selection = None
        self._mapping = self._scales = self.hierarchy = self.stage = None
        self._geometry_bindings = None
        self._gaussian_splats_bindings.clear()
        self._gaussian_splats_paths.clear()
        self._gaussian_splats_timestamp = -1


@configclass
class FabricBackendCfg(BackendCfg):
    """Native Fabric identity; the stage is borrowed from the active simulation."""

    class_type: type = FabricBackend
    stage: Usd.Stage = field(kw_only=True, metadata={"copy": False})
    device: str = field(kw_only=True)
