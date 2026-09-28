# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Persistent Gaussian geometry, full SH3, and material-frame bruise shading."""

import math
from types import SimpleNamespace

import numpy as np
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton.viewer import ViewerRTX

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from ._mpm.binding import make_binding
from ._mpm.rtx_gaussian_frame import GaussianLocalFrame
from ._mpm.rtx_sh_frame import SHMaterialFrame
from .render_settings import apply_sampling_settings, require_live_gaussian_renderer


class BerryViewer(ViewerRTX):
    """Render all prepared Gaussians without changing the MPM resolution."""

    def __init__(
        self,
        env,
        pipeline=True,
        partitions=1,
        antialiasing="default",
        rtpt_spp=None,
        **kwargs,
    ):
        require_live_gaussian_renderer()
        self.env = env
        self.berry = env.berry
        self.aperture = 0.08
        self.pipeline = pipeline
        self.antialiasing = antialiasing
        self.rtpt_spp = rtpt_spp
        self.sampling_overrides = {}
        self.pending_frame = None
        self.follow_berry = True
        self.reset_requested = False
        self.last_center = self.berry.sim.rest.mean(0).copy()
        self.path = "/World/Berry/Gaussians"
        self.indices = [np.arange(i, len(self.berry.asset["xyz"]), partitions) for i in range(partitions)]
        self.paths = [f"{self.path}/Part{i}" for i in range(partitions)]
        self.binding = make_binding(
            self.berry.asset,
            self.berry.proxy["xyz"],
            self.berry.proxy["regions"],
            self.berry.profile["simulation"]["binding"],
        )
        with wp.ScopedDevice("cuda:0"):
            self.binding.deform_gpu(self.berry.sim.x, host=False)
            self.frame = GaussianLocalFrame(self.berry.asset["xyz"], self.berry.asset["scales"], "cuda:0")
            lower = np.max([self.berry.asset["xyz"][idx].min(0) for idx in self.indices], axis=0)
            upper = np.min([self.berry.asset["xyz"][idx].max(0) for idx in self.indices], axis=0)
            self.frame.center, self.frame.half = (
                (lower + upper) / 2,
                (upper - lower) / 2,
            )
            if np.any(self.frame.half <= 0):
                raise ValueError("Partition bounds have no common volume")
            sh = SimpleNamespace(
                binding=self.binding,
                inverse=wp.array(np.linalg.inv(self.binding.base).astype(np.float32), dtype=wp.mat33),
                nearest=wp.array(self.binding.ids[:, 0].copy(), dtype=int),
            )
            self.sh_frame = SHMaterialFrame(sh, quaternion=True)
        super().__init__(**kwargs, environment="studio", fps=30, async_rendering=False)
        self.set_model(NewtonManager.get_model())
        eye = np.array([0.56, -0.025, 0.045])
        target = self.berry.offset + np.array([0, 0, 0.018])
        delta = target - eye
        self.set_camera(
            wp.vec3(*eye),
            math.degrees(math.asin(delta[2] / np.linalg.norm(delta))),
            math.degrees(math.atan2(delta[1], delta[0])),
        )
        self.camera.near = 0.0001
        self.camera.pivot = type(self.camera.pos)(*target)
        self.register_ui_callback(self.controls, position="side")

    def controls(self, ui):
        ui.text(f"{self.env.cfg.berry.title()} | continuous grasp")
        ui.text("LB enable | RT close | LT open | release to hold")
        ui.text("Keyboard: WASDQE / ZX TG CV; K close, J open; R reset")
        aperture = float(self.env.action_manager.get_term("gripper_action").processed_actions[0].sum())
        ui.text(f"Commanded aperture: {aperture * 1000:.1f} mm")
        ui.text(f"Peak finger reaction: {np.linalg.norm(self.berry.last_force, axis=1).max():.3f} N")
        _, self.follow_berry = ui.checkbox("Follow berry", self.follow_berry)
        if ui.button("Reset robot and berry"):
            self.reset_requested = True

    def _init_ovrtx(self):
        source = Usd.Stage.Open(self.berry.usd_stage.GetRootLayer().identifier)
        layer = source.Flatten()
        Sdf.CreatePrimInLayer(self.stage.GetRootLayer(), "/World")
        Sdf.CopySpec(layer, "/Berry", self.stage.GetRootLayer(), "/World/Berry")
        # The physics/authoring metadata is not part of the render scene.
        self.stage.RemovePrim("/World/Berry/TaskData")
        self.stage.RemovePrim("/World/Berry/Tissue")
        self.stage.RemovePrim(self.path)
        UsdGeom.Xform.Define(self.stage, self.path)
        original = source.GetPrimAtPath("/Berry/Gaussians")
        for path, idx in zip(self.paths, self.indices):
            Sdf.CopySpec(layer, "/Berry/Gaussians", self.stage.GetRootLayer(), path)
            prim = self.stage.GetPrimAtPath(path)
            for attr in prim.GetAttributes():
                if attr.GetName().startswith("berry:"):
                    prim.RemoveProperty(attr.GetName())
            for name, key, ctor in (
                ("positions", "xyz", Vt.Vec3fArray),
                ("scales", "scales", Vt.Vec3fArray),
                ("orientations", "rotations", Vt.QuatfArray),
                ("opacities", "alpha", Vt.FloatArray),
                ("radiance:sphericalHarmonicsCoefficients", "sh", Vt.Vec3fArray),
            ):
                array = self.berry.asset[key][idx]
                if key == "sh":
                    array = array.reshape(-1, 3)
                value = ctor.FromNumpy(np.ascontiguousarray(array, np.float32))
                attr = prim.GetAttribute(name)
                attr.Clear()
                attr.Set(value)
                if original.GetAttribute(name).GetNumTimeSamples():
                    attr.Set(value, 0)
                    attr.Set(value, 1)
            for name, ctor in (
                ("primvars:squishyShQuaternion", Vt.Vec4fArray),
                ("primvars:interior", Vt.IntArray),
            ):
                values = np.asarray(original.GetAttribute(name).Get())[idx]
                attr = prim.GetAttribute(name)
                attr.Clear()
                attr.Set(ctor.FromNumpy(np.ascontiguousarray(values)))
                if name.endswith("Quaternion"):
                    attr.Set(attr.Get(), 0)
                    attr.Set(attr.Get(), 1)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                UsdShade.Material.Get(self.stage, "/World/Berry/Materials/Radiance")
            )
        self._has_dynamic_gaussian_streaming = True
        super()._init_ovrtx()
        from ovrtx import BindingFlag

        # Persistent bindings establish OVRTX's animated-geometry dataflow.
        # By-name writes can read back correctly while leaving its BVH stale.
        self._berry_bindings = {
            name: self._rtx.bind_array_attribute(
                self.paths, name, dtype="float32", shape=(lanes,), flags=BindingFlag.OPTIMIZE
            )
            for name, lanes in (
                ("positions", 3),
                ("scales", 3),
                ("orientations", 4),
                ("primvars:squishyShQuaternion", 4),
            )
        }

    def _add_camera_lights_and_render_product(self):
        super()._add_camera_lights_and_render_product()
        for prim in self.stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                UsdGeom.Camera(prim).CreateClippingRangeAttr(Gf.Vec2f(0.0001, 100))
        product = self.stage.GetPrimAtPath(self._render_product_path)
        product.CreateAttribute("omni:rtx:dlss:frameGeneration", Sdf.ValueTypeNames.Bool).Set(False)
        product.RemoveProperty("omni:rtx:quality")
        product.RemoveProperty("omni:rtx:waitForEvents")
        self.sampling_overrides = apply_sampling_settings(product, self.antialiasing, self.rtpt_spp)

    def _prepare_berry(self):
        with wp.ScopedDevice("cuda:0"):
            xyz, scales, quats = self.binding.deform_gpu(self.berry.sim.x, host=False)
            xyz, scales, transform = self.frame.evaluate(xyz, scales)
            shading = self.sh_frame.evaluate(self.berry.sim.damage)
            values = {
                "positions": xyz.numpy(),
                "scales": scales.numpy(),
                "orientations": quats.numpy(),
                **{name: array.numpy() for name, array in shading.items()},
            }
        transform = transform.copy()
        transform[3, :3] += self.berry.offset
        return transform, values

    def _update_berry(self):
        from ovrtx import Semantic

        transform, values = self.prepared
        self._rtx.write_attribute(
            prim_paths=self.paths,
            attribute_name="omni:xform",
            tensor=np.repeat(transform[None], len(self.paths), axis=0),
            semantic=Semantic.XFORM_MAT4x4,
        )
        for name, value in values.items():
            self._berry_bindings[name].write([np.ascontiguousarray(value[idx]) for idx in self.indices])

    def _render_and_display(self):
        self._update_berry()
        from ovrtx import Device

        if not self.pipeline:
            self._render_products = self._rtx.step(render_products={self._render_product_path}, delta_time=1 / 30)
        if self._window is not None and self._window.context is not None:
            for product in (self._render_products or {}).values():
                for frame in product.frames:
                    for var in frame.render_vars.values():
                        if var.source_name == "LdrColor":
                            with var.map(device=Device.CUDA) as mapping:
                                pixels = wp.from_dlpack(mapping, dtype=wp.vec4ub)
                                self._blit_to_window(pixels)
                                mapping.unmap(stream=pixels.device.stream.cuda_stream)
        if self.pipeline:
            self.pending_frame = self._rtx.step_async(render_products={self._render_product_path}, delta_time=1 / 30)

    def capture_image(self):
        # OVRTX 0.6 keys render vars by prim path, not by source name.
        from ovrtx import Device

        self.finish_frame()
        for product in (self._render_products or {}).values():
            for frame in product.frames:
                for var in frame.render_vars.values():
                    if var.source_name == "LdrColor":
                        with var.map(device=Device.CPU) as mapping:
                            return np.from_dlpack(mapping).copy()
        raise RuntimeError(
            f"No color output: {[(str(k), str(v), str(v.frames)) for k, v in self._render_products.items()]}"
        )

    def save_screenshot(self, path):
        from PIL import Image

        Image.fromarray(self.capture_image()).save(path)

    def draw(self, time_s):
        self.prepared = self._prepare_berry()
        center = self.berry.sim.x.numpy().mean(0)
        if self.follow_berry:
            shift = center - self.last_center
            self.set_camera(wp.vec3(*(np.asarray(self.camera.pos) + shift)), self.camera.pitch, self.camera.yaw)
            self.camera.pivot += type(self.camera.pos)(*shift)
        self.last_center = center
        # Complete the previous frame before publishing any changed scene arrays.
        # The robot/MPM step overlapped that render using independent state buffers.
        self.finish_frame()
        self.begin_frame(time_s)
        self.log_state(NewtonManager.get_state())
        self.end_frame()
        if getattr(self, "gui", None) is not None:
            self.gui.update_camera_from_keys = lambda *args: None

    def finish_frame(self):
        if self.pending_frame is not None:
            self._render_products = self.pending_frame.wait().fetch()
            self.pending_frame = None

    def verify_geometry(self):
        """Read back the native renderer's complete published Gaussian arrays."""
        self.finish_frame()
        _, values = self.prepared
        for name, expected in values.items():
            restored = self._rtx.read_array_attribute(attribute_name=name, prim_paths=self.paths)
            for path, idx in zip(self.paths, self.indices):
                np.testing.assert_allclose(
                    np.from_dlpack(restored[path]).reshape(expected[idx].shape),
                    expected[idx],
                    atol=1e-7,
                )
        for name, key in (
            ("opacities", "alpha"),
            ("radiance:sphericalHarmonicsCoefficients", "sh"),
        ):
            restored = self._rtx.read_array_attribute(attribute_name=name, prim_paths=self.paths)
            for path, idx in zip(self.paths, self.indices):
                expected = self.berry.asset[key][idx]
                np.testing.assert_array_equal(np.from_dlpack(restored[path]).reshape(expected.shape), expected)
        return True

    def close(self):
        self.finish_frame()
        renderer = self._rtx
        for binding in getattr(self, "_berry_bindings", {}).values():
            binding.unbind()
        super().close()
        if renderer is not None:
            renderer.destroy()
