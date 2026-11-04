# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish tissue-driven Gaussian deformation through shared scene data."""

from isaaclab.scene_data import GaussianField

from .mpm_binding import MPMBinding
from .vbd_binding import VBDBinding


def publish_gaussians(tissues, provider):
    """Allocate persistent deformation bindings and publish renderer-independent fields."""
    for name, berry in tissues.items():
        particles = berry.particles
        binding = (
            VBDBinding(berry.gaussians, particles["xyz"], particles["tets"], berry.device)
            if "tets" in particles
            else MPMBinding(berry.gaussians, particles["xyz"], particles["regions"], berry.device)
        )

        def update(berry=berry, binding=binding):
            binding.deform(berry.positions_warp(), berry.damage)

        provider.register_gaussian_field(
            f"/World/Visuals/Berries/{name}/Gaussians/Field",
            GaussianField(binding.positions, binding.orientations, binding.scales, update=update),
        )
