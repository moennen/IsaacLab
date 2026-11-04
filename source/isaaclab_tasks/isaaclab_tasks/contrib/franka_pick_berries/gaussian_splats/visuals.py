# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish tissue-driven Gaussian deformation through shared scene data."""

from isaaclab.scene_data import GaussianField

from .mpm_binding import MPMBinding
from .vbd_binding import VBDBinding


def gaussian_binding(tissue):
    """Bind Gaussian appearance to the tissue solver's native deformation representation."""
    particles = tissue.particles
    if "simplicits" in particles:
        from .simplicits_binding import SimplicitsBinding

        return SimplicitsBinding(tissue)
    if "tets" in particles:
        return VBDBinding(tissue.gaussians, particles["xyz"], particles["tets"], tissue.device)
    return MPMBinding(tissue.gaussians, particles["xyz"], particles["regions"], tissue.device)


def publish_gaussians(tissues, provider):
    """Allocate persistent deformation bindings and publish renderer-independent fields."""
    for name, berry in tissues.items():
        binding = gaussian_binding(berry)

        def update(berry=berry, binding=binding):
            binding.deform(berry.positions_warp(), berry.damage)

        provider.register_gaussian_field(
            f"/World/Visuals/Berries/{name}/Gaussians/Field",
            GaussianField(binding.positions, binding.orientations, binding.scales, update=update),
        )
