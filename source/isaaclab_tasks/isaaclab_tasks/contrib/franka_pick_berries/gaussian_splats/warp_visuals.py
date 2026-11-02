# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deforming berry Gaussian shapes for Newton's Warp camera renderer."""

import copy

import newton
import warp as wp

from .mpm_binding import MPMBinding
from .render_delegate import _bounds
from .vbd_binding import VBDBinding


@wp.kernel
def _write_transforms(
    positions: wp.array[wp.vec3], orientations: wp.array[wp.quat], offset: wp.vec3, transforms: wp.array[wp.transform]
):
    i = wp.tid()
    transforms[i] = wp.transform(positions[i] + offset, orientations[i])


@wp.kernel
def _init_bounds(bounds: wp.array2d[float]):
    axis = wp.tid()
    bounds[0, axis] = wp.inf
    bounds[1, axis] = -wp.inf


@wp.kernel
def _publish_bounds(bounds: wp.array2d[float], offset: wp.vec3, shape: int, shape_bounds: wp.array2d[wp.vec3]):
    # Four sigma conservatively contains the 0.01 response cutoff, including rotations.
    center = wp.vec3(bounds[0, 0], bounds[0, 1], bounds[0, 2])
    upper = wp.vec3(bounds[1, 0], bounds[1, 1], bounds[1, 2])
    middle = (center + upper) * 0.5
    half = (upper - center) * (4.0 / 6.0)
    shape_bounds[shape, 0] = middle - half + offset
    shape_bounds[shape, 1] = middle + half + offset


def import_scene_visuals(builder, specs):
    """Consume the shared USD scene through Newton's standard visual importer."""
    from pxr import Usd, UsdGeom

    from ..scene.visuals import author_scene_visuals

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    author_scene_visuals(stage, specs)
    start = builder.shape_count
    builder.add_usd(stage, root_path="/World", load_visual_shapes=True, skip_mesh_approximation=True, load_sites=False)
    for shape in range(start, builder.shape_count):
        builder.shape_flags[shape] &= int(newton.ShapeFlags.VISIBLE)


class WarpGaussianVisuals:
    """Keep finalized Gaussian buffers and BVHs consistent with the simulated tissue."""

    def __init__(self, env):
        self.env = env
        model = env.sim.backend_manager.backend.model
        self.entries = []
        sources = {
            label: (i, source)
            for i, (label, source) in enumerate(zip(model.shape_label, model.shape_source, strict=True))
        }
        for name, berry in env.berries.items():
            shape, gaussian = sources[f"/World/Berries/{name}/Gaussians/Field"]
            particles = berry.particles
            binding = (
                VBDBinding(berry.gaussians, particles["xyz"], particles["tets"], berry.device)
                if "tets" in particles
                else MPMBinding(berry.gaussians, particles["xyz"], particles["regions"], berry.device)
            )
            self.entries.append((berry, gaussian, binding, shape, wp.empty((2, 3), dtype=float, device=berry.device)))

    def update(self, _event=None):
        """Deform the splats and refit both Gaussian and scene acceleration structures."""
        for berry, gaussian, binding, shape, bounds in self.entries:
            with wp.ScopedDevice(berry.device):
                binding.deform(berry.positions_warp(), berry.damage)
                data = gaussian._warp_data
                wp.launch(
                    _write_transforms,
                    dim=len(binding.positions),
                    inputs=[binding.positions, binding.orientations, wp.vec3(0.0), data.transforms],
                )
                wp.copy(data.scales, binding.scales)
                gaussian.bvh_refit()
                wp.launch(_init_bounds, dim=3, inputs=[bounds])
                wp.launch(_bounds, dim=len(binding.positions), inputs=[binding.positions, binding.scales, bounds])
                wp.launch(
                    _publish_bounds,
                    dim=1,
                    inputs=[
                        bounds,
                        wp.vec3(0.0),
                        shape,
                        self.env.sim.backend_manager.backend.model.bvh_shape_bounds,
                    ],
                )
        backend = self.env.sim.backend_manager.backend
        # Gaussian local bounds change with deformation, unlike rigid shape bounds.
        backend.model.bvh_refit_shapes(backend.state_0)


def configure_camera(camera, visuals):
    """Render splats instead of their simulation particles or tetrahedral surface.

    A shallow render-only model view preserves all GPU shape/state buffers. Its empty triangle topology hides the
    VBD simulation surface without changing the model used by physics.
    """
    renderer = camera._renderer
    sensor = renderer.newton_sensor
    if sensor.model.tri_count:
        model = copy.copy(sensor.model)
        model.tri_count = 0
        model.tri_indices = wp.empty(0, dtype=wp.vec3i, device=model.device)
        sensor = newton.sensors.SensorTiledCamera(model, default_render_config=copy.copy(sensor.default_render_config))
        if renderer.cfg.create_default_light:
            sensor.utils.create_default_light(enable_shadows=renderer.cfg.enable_shadows)
        renderer.newton_sensor = sensor
    camera._render_data.newton_sensor = sensor
    camera._render_data.graph = None
    sensor.default_render_config.enable_particles = False
    sensor.default_render_config.gaussians_mode = newton.sensors.SensorTiledCamera.GaussianRenderMode.QUALITY
    sensor.default_render_config.gaussians_min_transmittance = 0.01
    sensor.default_render_config.gaussians_max_num_hits = 64

    prepare_camera = camera._prepare_camera

    def prepare_with_gaussians(mask):
        visuals.update()
        prepare_camera(mask)

    camera._prepare_camera = prepare_with_gaussians
