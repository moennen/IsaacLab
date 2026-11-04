# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive RTX viewer of the workcell: Newton's robot and props, the deforming berries and the scanned room."""

import copy
import math
import time

import newton
import numpy as np
import ovstage
import warp as wp
from newton.viewer import ViewerRTX

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from isaaclab.sim import get_current_stage

from ..gaussian_splats.render_delegate import GaussianRenderDelegate
from ..scene.visuals import VISUAL_ROOT_PATH
from ..scene.workcell_materials import restore_workcell_materials
from .camera import VERTICAL_FOV_DEG, camera_view

CAMERA_VIEWS = ("punnet_and_bowl", "berry_closeup", "room", "scripted_camera")
"""Camera views: the punnet and bowl, a close-up following the handled berry, the whole room, or a camera moved by a
scripted camera such as :class:`~.scripted_camera_demo.ScriptedCameraDemo`."""


class BerryViewer(ViewerRTX):
    """Newton's RTX viewer, extended with the berries' Gaussians, the tableware and the room.

    On every :meth:`draw`, each berry's Gaussians follow its tissue
    (:class:`..gaussian_splats.render_delegate.GaussianRenderDelegate`), written to the renderer's OVStage scene with
    the rest of the viewer's updates. Rendering overlaps the next physics step: a frame is submitted asynchronously and
    collected before the next one changes the scene.

    Args:
        env: The berry environment.
        samples_per_pixel: Real-time path-tracing samples per pixel; None keeps the renderer's default.
        camera: Initial camera view, one of :data:`CAMERA_VIEWS`.
        f_stop: Camera f-number for depth of field in a scripted camera's close shots; None keeps everything sharp.
        stochastic_gaussian_rendering: Shade the berries with their own material, which turns their colour with the
            tissue and tints bruises, but is stochastically composited, with visible sampling noise between frames.
            Otherwise the Gaussians use the renderer's built-in ParticleFieldEmissive material, composited
            deterministically for camera rays.
        **kwargs: Arguments of :class:`newton.viewer.ViewerRTX`, such as ``width``, ``height`` and ``headless``.
    """

    def __init__(
        self,
        env,
        samples_per_pixel=None,
        camera="punnet_and_bowl",
        f_stop=None,
        stochastic_gaussian_rendering=False,
        **kwargs,
    ):
        if camera not in CAMERA_VIEWS:
            raise ValueError(f"Unknown camera view {camera!r}; expected one of {CAMERA_VIEWS}")
        self.env = env
        self.handled_berry = env.handled_berry
        self.samples_per_pixel = samples_per_pixel
        self.f_stop = f_stop
        self.stochastic_gaussian_rendering = stochastic_gaussian_rendering
        # Keyboard teleoperation's commanded aperture [m].
        self.keyboard_aperture = 0.08
        self.pending_frame = None
        self._prepared = []
        # Camera lens attributes [mm or f-number] to write with the next scene update.
        self._pending_lens = {}
        self.camera_follows_berry = False
        # Set by the reset button or the R key; the caller resets and clears it.
        self.reset_requested = False
        self._reset_key = False
        # Frames drawn per second of wall-clock time, smoothed.
        self.frame_rate = 0.0
        self._last_draw = None
        # Phase of a scripted sequence, shown in the controls.
        self.status_text = None
        self.last_center = self.handled_berry.positions().mean(0)
        self.render_delegates = [
            GaussianRenderDelegate(
                berry, f"{VISUAL_ROOT_PATH}/Berries/{name}", builtin_material=not stochastic_gaussian_rendering
            )
            for name, berry in env.berries.items()
        ]
        super().__init__(**kwargs, environment="studio", fps=30, async_rendering=False)
        self.camera.fov = VERTICAL_FOV_DEG
        # The shared USD asset is copied below with its original materials. Hide its native
        # representation in this legacy viewer's model layer to avoid drawing it twice.
        model = copy.copy(self._newton.model)
        flags = model.shape_flags.numpy()
        for index, label in enumerate(model.shape_label):
            if label.startswith(VISUAL_ROOT_PATH + "/"):
                flags[index] &= ~int(newton.ShapeFlags.VISIBLE)
        model.shape_flags = wp.array(flags, dtype=model.shape_flags.dtype, device=model.device)
        self.set_model(model)
        # The berries appear as their Gaussians only, not as the VBD solver's tetrahedral mesh.
        self.show_triangles = False
        # A scripted camera starts on the overview and is then moved with place_camera().
        self.set_camera_view("punnet_and_bowl" if camera == "scripted_camera" else camera)
        self.register_ui_callback(self.side_panel, position="side")

    @property
    def _newton(self):
        """The Newton backend of the environment's simulation."""
        return self.env.sim.backend_manager.backend

    def set_camera_view(self, view: str) -> None:
        """Move the camera to a fixed view; the berry view then follows the handled berry."""
        eye, target = camera_view(view, self.handled_berry)
        self.camera_follows_berry = view == "berry_closeup"
        self._look(eye, target)

    def set_handled_berry(self, berry) -> None:
        """Make ``berry`` the one the close-up follows."""
        if berry is not self.handled_berry:
            self.handled_berry = berry
            self.last_center = berry.positions().mean(0)

    def place_camera(
        self,
        eye: np.ndarray,
        target: np.ndarray,
        fov: float | None = None,
        focus: float | None = None,
        depth_of_field: bool = True,
    ) -> None:
        """Place the camera at ``eye`` looking at ``target`` [m], for example from a scripted camera.

        Args:
            eye: Camera position [m].
            target: Point the camera looks at [m].
            fov: Vertical field of view [deg]; None keeps it.
            focus: Focus distance [m] for depth of field (see ``f_stop``); None keeps it.
            depth_of_field: Whether this view uses the depth of field of ``f_stop``, or is sharp throughout.
        """
        self._look(np.asarray(eye, float), np.asarray(target, float))
        if fov is not None:
            self.camera.fov = fov
            # As in Newton's viewer: the vertical aperture is 20.955 mm.
            self._write_camera("focalLength", 20.955 / (2.0 * math.tan(math.radians(fov) / 2.0)))
        if self.f_stop:
            # An f-number of 0 turns depth of field off.
            self._write_camera("fStop", float(self.f_stop) if depth_of_field else 0.0)
            if focus is not None:
                self._write_camera("focusDistance", focus)

    def side_panel(self, ui):
        """Side panel: status, help, camera views and reset."""
        count = len(self.env.berries)
        ui.text(f"{count} raspberr{'ies' if count > 1 else 'y'} | continuous grasp")
        if self.status_text:
            ui.text(self.status_text)
        ui.text("Gamepad: hold LB; sticks move, RT closes, LT opens")
        ui.text("Keyboard: WASDQE move, ZX TG CV turn, K close, J open, R reset")
        aperture = float(self.env.action_manager.get_term("gripper_action").processed_actions[0].sum())
        ui.text(f"Commanded aperture: {aperture * 1000:.1f} mm")
        ui.text(f"Frame rate: {self.frame_rate:.1f} frames/s")
        _, self.camera_follows_berry = ui.checkbox("Follow berry", self.camera_follows_berry)
        if ui.button("Room view"):
            self.set_camera_view("room")
        if ui.button("Punnet and bowl view"):
            self.set_camera_view("punnet_and_bowl")
        for name, berry in self.env.berries.items():
            if ui.button(f"Close-up of {name}"):
                self.set_handled_berry(berry)
                self.set_camera_view("berry_closeup")
        if ui.button("Reset robot and berries"):
            self.reset_requested = True

    def draw(self, time_s: float) -> None:
        """Deform the berries' Gaussians and render one frame, while the next physics step runs."""
        now = time.perf_counter()
        if self._last_draw is not None:
            self.frame_rate = 0.9 * self.frame_rate + 0.1 / max(now - self._last_draw, 1e-6)
        self._last_draw = now
        key = self.is_key_down("R")
        self.reset_requested |= key and not self._reset_key
        self._reset_key = key
        self._prepared = [delegate.deform() for delegate in self.render_delegates]
        center = self.handled_berry.positions().mean(0)
        if self.camera_follows_berry:
            shift = center - self.last_center
            self.set_camera(wp.vec3(*(np.asarray(self.camera.pos) + shift)), self.camera.pitch, self.camera.yaw)
            self.camera.pivot += type(self.camera.pos)(*shift)
        self.last_center = center
        # The previous frame must finish before the scene changes; the physics step overlapped it.
        self.finish_frame()
        self.begin_frame(time_s)
        self.log_state(self._newton.state_0)
        self.end_frame()
        if getattr(self, "gui", None) is not None:
            # Keyboard keys drive the robot, not the camera.
            self.gui.update_camera_from_keys = lambda *args: None

    def finish_frame(self) -> None:
        """Wait for the frame in flight, if any."""
        if self.pending_frame is not None:
            self._render_products = self.pending_frame.wait().fetch()
            self.pending_frame = None

    def capture_image(self) -> np.ndarray:
        """Return the last rendered frame as an (height, width, 4) RGBA image."""
        from ovrtx import Device

        self.finish_frame()
        for var in self._color_outputs():
            with var.map(device=Device.CPU) as mapping:
                return np.from_dlpack(mapping).copy()
        raise RuntimeError("The renderer produced no color output")

    def close(self):
        self.finish_frame()
        renderer = self._rtx
        super().close()
        if renderer is not None:
            renderer.destroy()

    # ----------------------------------------------------------------------------------------------------------------
    # Scene authoring and rendering hooks of ViewerRTX
    # ----------------------------------------------------------------------------------------------------------------

    def _init_ovrtx(self):
        restore_workcell_materials(
            self.stage,
            get_current_stage(),
            (
                (f"{self._get_path(batch.name)}/instance_{index}", self.model.shape_label[shape])
                for batch in self._shape_instances.values()
                for index, shape in enumerate(batch.model_shapes)
            ),
        )
        Sdf.CopySpec(get_current_stage().Flatten(), VISUAL_ROOT_PATH, self.stage.GetRootLayer(), VISUAL_ROOT_PATH)
        if self.stochastic_gaussian_rendering:
            for delegate in self.render_delegates:
                original = UsdShade.Shader.Get(delegate.tissue.stage, "/Berry/Materials/Radiance/Shader")
                shader = UsdShade.Shader.Get(self.stage, delegate.shader_path)
                asset = original.GetSourceAsset("mdl")
                shader.SetSourceAsset(Sdf.AssetPath(asset.resolvedPath or asset.path), "mdl")
                shader.SetSourceAssetSubIdentifier(original.GetSourceAssetSubIdentifier("mdl"), "mdl")
        # Newton loads the stage with OVStage 0.2's ``open_usd(..., time_code=)``, which OVStage 0.3 names ``time``.
        open_usd = ovstage.population.open_usd

        def open_usd_at_time_code(stage, path, ordinal, time_code=math.nan, **kwargs):
            return open_usd(stage, path, ordinal, time=time_code, **kwargs)

        ovstage.population.open_usd = open_usd_at_time_code
        try:
            super()._init_ovrtx()
        finally:
            ovstage.population.open_usd = open_usd

    def _add_studio_lights(self):
        # The scan already contains the room's illumination. Light the meshes with broad, warm indoor sources.
        dome = UsdLux.DomeLight.Define(self.stage, "/World/RoomAmbient")
        dome.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.93))
        dome.CreateIntensityAttr(350.0)
        for name, position, size, intensity in (
            ("RoomKey", (-0.8, -0.6, 2.4), (2.0, 1.5), 4800.0),
            ("RoomFill", (0.5, 1.0, 2.6), (2.0, 2.0), 1400.0),
        ):
            light = UsdLux.RectLight.Define(self.stage, f"/World/{name}")
            light.CreateWidthAttr(size[0])
            light.CreateHeightAttr(size[1])
            light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.93))
            light.CreateIntensityAttr(intensity)
            # USD area lights emit along local -Z; aim at the work surface.
            transform = Gf.Matrix4d().SetRotate(
                Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(0.45, 0, 0) - Gf.Vec3d(*position))
            )
            transform.SetTranslateOnly(Gf.Vec3d(*position))
            UsdGeom.Xformable(light).AddTransformOp().Set(transform)

    def _add_camera_lights_and_render_product(self):
        super()._add_camera_lights_and_render_product()
        for prim in self.stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                camera = UsdGeom.Camera(prim)
                camera.CreateClippingRangeAttr(Gf.Vec2f(0.0001, 100))
                if self.f_stop:
                    camera.CreateFStopAttr(float(self.f_stop))
                    camera.CreateFocusDistanceAttr(0.5)
        product = self.stage.GetPrimAtPath(self._render_product_path)
        product.CreateAttribute("omni:rtx:dlss:frameGeneration", Sdf.ValueTypeNames.Bool).Set(False)
        product.RemoveProperty("omni:rtx:quality")
        product.RemoveProperty("omni:rtx:waitForEvents")
        if self.samples_per_pixel is not None:
            product.CreateAttribute("omni:rtx:rtpt:spp", Sdf.ValueTypeNames.Int).Set(self.samples_per_pixel)
        if not self.stochastic_gaussian_rendering:
            product.CreateAttribute("omni:rtx:rtpt:gaussian:nonStochastic:enabled", Sdf.ValueTypeNames.Bool).Set(True)

    def _update_ovrtx_camera(self):
        super()._update_ovrtx_camera()
        for attribute, value in self._pending_lens.items():
            self._write_runtime_attribute([self._camera_prim_path], attribute, np.array([value], np.float32))
        self._pending_lens.clear()

    def _update_ovrtx_mesh_points(self):
        # The deformed Gaussians are written with the scene's other per-frame geometry.
        super()._update_ovrtx_mesh_points()
        for delegate, prepared in zip(self.render_delegates, self._prepared):
            delegate.publish(prepared, self._write_runtime_attribute)
        self._prepared = []

    def _render_and_display(self):
        # Show the frame that just finished and submit the next one, of the updated scene, without waiting.
        from ovrtx import Device

        if self._window is not None and self._window.context is not None:
            for var in self._color_outputs():
                with var.map(device=Device.CUDA) as mapping:
                    pixels = wp.from_dlpack(mapping, dtype=wp.vec4ub)
                    self._blit_to_window(pixels)
                    mapping.unmap(stream=pixels.device.stream.cuda_stream)
            # The blit only fills the presentation texture; drawing it, with the UI, swaps the window's buffers.
            frame = self._displayed_frame
            self._present(frame.texture or None, frame.width, frame.height)
        self.pending_frame = self._rtx.step_async(
            render_products={self._render_product_path}, delta_time=1 / 30, ordinal=self._ovstage_ordinal
        )

    def _color_outputs(self):
        # OVRTX 0.6 keys render variables by prim path, not by source name.
        for product in (self._render_products or {}).values():
            for frame in product.frames:
                for var in frame.render_vars.values():
                    if var.source_name == "LdrColor":
                        yield var

    def _write_camera(self, attribute: str, value: float) -> None:
        # Before the renderer starts, the authored camera applies.
        if self._rtx is not None:
            self._pending_lens[attribute] = value

    def _look(self, eye: np.ndarray, target: np.ndarray) -> None:
        delta = target - eye
        self.set_camera(
            wp.vec3(*eye),
            math.degrees(math.asin(delta[2] / np.linalg.norm(delta))),
            math.degrees(math.atan2(delta[1], delta[0])),
        )
        self.camera.near = 0.0001
        self.camera.pivot = type(self.camera.pos)(*target)
