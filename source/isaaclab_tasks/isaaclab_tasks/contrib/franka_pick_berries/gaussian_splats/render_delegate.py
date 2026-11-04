# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish one berry's deforming Gaussians to the RTX renderer on every frame, without leaving the GPU.

The berry's Gaussians are authored once in the render stage, from its asset. On every frame
:meth:`GaussianRenderDelegate.deform` deforms and shades them from the tissue particles (:mod:`.mpm_binding`), or
from the nodes of the tissue's tetrahedral mesh with the VBD solver (:mod:`.vbd_binding`), and
:meth:`GaussianRenderDelegate.publish` writes the resulting GPU arrays to the renderer's OVStage scene. Only the
per-frame attributes change: positions, scales, orientations and, with the berry's own material, the shading
quaternion; opacities and spherical harmonics stay as authored.
"""

import numpy as np
import warp as wp

from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

from .visuals import gaussian_binding

SHADING = "primvars:squishyShQuaternion"
"""Per-Gaussian SH rotation and bruise tint read by the berry's MDL shader."""

# Per-frame Gaussian attributes and their number of float lanes.
_DYNAMIC = (("positions", 3), ("scales", 3), ("orientations", 4), (SHADING, 4))


@wp.kernel
def _bounds(x: wp.array[wp.vec3], s: wp.array[wp.vec3], bounds: wp.array2d[float]):
    i = wp.tid()
    reach = 3.0 * wp.max(s[i][0], wp.max(s[i][1], s[i][2]))
    for axis in range(3):
        wp.atomic_min(bounds, 0, axis, x[i][axis] - reach)
        wp.atomic_max(bounds, 1, axis, x[i][axis] + reach)


@wp.kernel
def _normalize(
    x: wp.array[wp.vec3],
    s: wp.array[wp.vec3],
    center: wp.vec3,
    rest_center: wp.vec3,
    factor: float,
    local_x: wp.array[wp.vec3],
    local_s: wp.array[wp.vec3],
):
    i = wp.tid()
    local_x[i] = (x[i] - center) / factor + rest_center
    local_s[i] = s[i] / factor


def author_berry_visual(stage: Usd.Stage, tissue, root_path: str, builtin_material: bool = False) -> None:
    """Author a berry's placed Gaussian field and material from its asset."""
    field_path = f"{root_path}/Gaussians/Field"
    shader_path = f"{root_path}/Materials/Radiance/Shader"
    source = tissue.stage
    layer = source.Flatten()
    Sdf.CreatePrimInLayer(stage.GetRootLayer(), root_path)
    Sdf.CopySpec(layer, "/Berry", stage.GetRootLayer(), root_path)
    # The physics data is not part of the render scene.
    stage.RemovePrim(f"{root_path}/TaskData")
    stage.RemovePrim(f"{root_path}/Tissue")
    stage.RemovePrim(f"{root_path}/SimulationMesh")
    # The renderer only loads a Gaussian field copied on its own under a defined transform, not one copied as
    # part of the asset's tree.
    stage.RemovePrim(f"{root_path}/Gaussians")
    UsdGeom.Xform.Define(stage, f"{root_path}/Gaussians")
    Sdf.CopySpec(layer, "/Berry/Gaussians", stage.GetRootLayer(), field_path)
    prim = stage.GetPrimAtPath(field_path)
    for attr in prim.GetAttributes():
        if attr.GetName().startswith(("berry:", "kaolin_skinned_physics:", "kaolin_physics_material:")):
            prim.RemoveProperty(attr.GetName())
    schemas = prim.GetMetadata("apiSchemas")
    if schemas:
        for schema in schemas.GetAppliedItems():
            if schema.startswith(("KaolinSkinnedPhysicsAPI:", "KaolinPhysicsMaterialAPI:")):
                prim.RemoveAppliedSchema(schema)
    # The berry may have been turned and moved into the punnet: author its placed Gaussians. Time samples mark the
    # attributes the renderer should expect to change.
    original = source.GetPrimAtPath("/Berry/Gaussians")
    for name, key, array_type in (
        ("positions", "xyz", Vt.Vec3fArray),
        ("scales", "scales", Vt.Vec3fArray),
        ("orientations", "rotations", Vt.QuatfArray),
        ("opacities", "alpha", Vt.FloatArray),
        ("radiance:sphericalHarmonicsCoefficients", "sh", Vt.Vec3fArray),
    ):
        array = tissue.gaussians[key].reshape(-1, 3) if key == "sh" else tissue.gaussians[key]
        value = array_type.FromNumpy(np.ascontiguousarray(array, np.float32))
        attr = prim.GetAttribute(name)
        attr.Clear()
        attr.Set(value)
        if original.GetAttribute(name).GetNumTimeSamples():
            attr.Set(value, 0)
            attr.Set(value, 1)
    shading = prim.GetAttribute(SHADING)
    value = shading.Get()
    shading.Clear()
    for time in (Usd.TimeCode.Default(), 0, 1):
        shading.Set(value, time)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(stage, f"{root_path}/Materials/Radiance"))
    if builtin_material:
        shader = UsdShade.Shader.Get(stage, shader_path)
        shader.SetSourceAsset(Sdf.AssetPath("ParticleFieldEmissive.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("ParticleFieldEmissive", "mdl")


class GaussianRenderDelegate:
    """Author one berry's Gaussians in the render stage and publish their deformation on every frame.

    Args:
        tissue: The berry's tissue (:class:`~..physics.tissue.BerryTissue`).
        root_path: Render-stage path under which the berry's Gaussians and material are authored.
        builtin_material: Shade the Gaussians with the renderer's built-in ParticleFieldEmissive material instead of the
            berry's own, which turns their colour with the tissue and tints bruises from the shading quaternion.
    """

    def __init__(self, tissue, root_path: str, builtin_material: bool = False):
        self.tissue = tissue
        self.root_path = root_path
        self.path = f"{root_path}/Gaussians/Field"
        self.shader_path = f"{root_path}/Materials/Radiance/Shader"
        self.builtin_material = builtin_material
        # The built-in material ignores the shading quaternion, which is then not published.
        self._dynamic = [(name, lanes) for name, lanes in _DYNAMIC if not (builtin_material and name == SHADING)]
        gaussians = tissue.gaussians
        device = tissue.device
        self.binding = gaussian_binding(tissue)
        # The renderer keeps the Gaussians' bounds from their authored extent. The deformed Gaussians are therefore
        # published in a local frame that keeps them inside their rest bounds, with a transform that moves them back
        # into place.
        self._rest_center = (gaussians["xyz"].min(0) + gaussians["xyz"].max(0)) / 2
        self._rest_half = (gaussians["xyz"].max(0) - gaussians["xyz"].min(0)) / 2
        count = len(gaussians["xyz"])
        with wp.ScopedDevice(device):
            self._extent = wp.empty((2, 3), dtype=float)
            # The stage copies the published arrays when they are written, so one set is refilled every frame.
            dtypes = {"positions": wp.vec3, "scales": wp.vec3, "orientations": wp.quat, SHADING: wp.vec4}
            self._buffers = {name: wp.empty(count, dtype=dtypes[name]) for name, _ in self._dynamic}

    def author_in_stage(self, stage: Usd.Stage) -> None:
        """Copy the berry's Gaussians and material from its asset into the render stage."""
        author_berry_visual(stage, self.tissue, self.root_path, self.builtin_material)

    def deform(self) -> tuple[np.ndarray, dict]:
        """Deform and shade the Gaussians on the GPU; return their transform and the arrays to publish."""
        binding = self.binding
        with wp.ScopedDevice(self.tissue.device):
            binding.deform(self.tissue.positions_warp(), self.tissue.damage)
            out = self._buffers
            # Local frame: center the deformed Gaussians on their rest center and shrink them into their rest bounds.
            self._extent.assign(np.array([[np.inf] * 3, [-np.inf] * 3], np.float32))
            wp.launch(_bounds, dim=len(binding.positions), inputs=[binding.positions, binding.scales, self._extent])
            extent = self._extent.numpy().astype(np.float64)
            center = extent.mean(0)
            factor = float(np.max((extent[1] - extent[0]) / 2 / self._rest_half))
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError("Invalid deformed Gaussian bounds")
            wp.launch(
                _normalize,
                dim=len(binding.positions),
                inputs=[binding.positions, binding.scales, wp.vec3(*center), wp.vec3(*self._rest_center), factor],
                outputs=[out["positions"], out["scales"]],
            )
            wp.copy(out["orientations"], binding.orientations)
            if SHADING in out:
                wp.copy(out[SHADING], binding.shading)
        # Row-vector convention: scale, then translate back to the deformed center and to the berry's place.
        transform = np.eye(4)
        transform[:3, :3] *= factor
        transform[3, :3] = center - factor * self._rest_center + self.tissue.offset
        return transform, out

    def publish(self, prepared: tuple[np.ndarray, dict], write) -> None:
        """Publish arrays from :meth:`deform` to the renderer's stage.

        Args:
            prepared: The transform and arrays returned by :meth:`deform`.
            write: The viewer's stage writer, called as ``write(prim_paths, attribute_name, values, **flags)`` while
                the viewer updates its scene.
        """
        import ovstage

        transform, values = prepared
        write([self.path], "omni:xform", transform[None], is_matrix=True)
        cuda_stream = wp.get_stream(self.tissue.device).cuda_stream
        lanes = dict(self._dynamic)
        for name, array in values.items():
            # A GPU array of float vectors, which the stage copies after the kernels on its stream.
            tensor = ovstage.make_dltensor(
                array, dtype=ovstage.numpy_to_dldatatype(np.dtype(np.float32), lanes=lanes[name]), shape=[len(array)]
            )
            write([self.path], name, [tensor], is_array=True, cuda_stream=cuda_stream)
