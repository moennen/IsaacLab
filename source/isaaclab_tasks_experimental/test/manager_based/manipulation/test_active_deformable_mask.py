# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU unit test for the play-only active-deformable masking kernel.

Runs Warp on the CPU device so it exercises the mask logic without a GPU sim.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.mdp.active_deformable import (
    gather_compact_particles_vec3f,
    segmented_max,
    segmented_min,
    set_active_mask_flags,
)


def test_set_active_mask_flags_activates_only_the_chosen_object():
    wp.init()
    device = "cpu"
    num_envs = 3
    particles_per_body = 4
    num_objects = 2
    total_particles = num_objects * num_envs * particles_per_body

    # Object i owns a contiguous per-env block in the shared flat particle array.
    offsets = np.array([[0, 4, 8], [12, 16, 20]], dtype=np.int32)  # (num_objects, num_envs)
    # env 0 -> object 1, env 1 -> object 0, env 2 -> object 1.
    active_index = np.array([1, 0, 1], dtype=np.int32)

    env_ids = wp.array(np.arange(num_envs, dtype=np.int32), device=device)
    active_index_wp = wp.array(active_index, device=device)

    # Distinct sentinel pristine values so "restored to pristine" is unambiguous.
    pristine_flag_value = 1
    pristine_inv_mass_value = 0.5

    out_flags_np = np.full(total_particles, 99, dtype=np.int32)
    out_inv_mass_np = np.full(total_particles, 99.0, dtype=np.float32)
    model_flags_np = np.full(total_particles, 99, dtype=np.int32)
    model_inv_mass_np = np.full(total_particles, 99.0, dtype=np.float32)
    out_flags = wp.array(out_flags_np, device=device)
    out_inv_mass = wp.array(out_inv_mass_np, device=device)
    model_flags = wp.array(model_flags_np, device=device)
    model_inv_mass = wp.array(model_inv_mass_np, device=device)

    for i in range(num_objects):
        pristine_flags = wp.array(np.full(total_particles, pristine_flag_value, dtype=np.int32), device=device)
        pristine_inv_mass = wp.array(np.full(total_particles, pristine_inv_mass_value, dtype=np.float32), device=device)
        wp.launch(
            set_active_mask_flags,
            dim=(num_envs, particles_per_body),
            inputs=[
                env_ids,
                active_index_wp,
                i,
                wp.array(offsets[i], device=device),
                pristine_flags,
                pristine_inv_mass,
                out_flags,
                out_inv_mass,
                model_flags,
                model_inv_mass,
            ],
            device=device,
        )

    out_flags_np = out_flags.numpy()
    out_inv_mass_np = out_inv_mass.numpy()
    model_flags_np = model_flags.numpy()
    model_inv_mass_np = model_inv_mass.numpy()

    for i in range(num_objects):
        for e in range(num_envs):
            base = int(offsets[i, e])
            body = slice(base, base + particles_per_body)
            if active_index[e] == i:
                assert np.all(out_flags_np[body] == pristine_flag_value)
                assert np.allclose(out_inv_mass_np[body], pristine_inv_mass_value)
                assert np.all(model_flags_np[body] == pristine_flag_value)
                assert np.allclose(model_inv_mass_np[body], pristine_inv_mass_value)
            else:
                assert np.all(out_flags_np[body] == 0)
                assert np.allclose(out_inv_mass_np[body], 0.0)
                assert np.all(model_flags_np[body] == 0)
                assert np.allclose(model_inv_mass_np[body], 0.0)


def test_active_offset_gather_matches_selected_object():
    """The proxy's active-offset gather must select each env's active-object offset."""
    import torch

    offsets = torch.tensor([[0, 4, 8], [12, 16, 20]], dtype=torch.int32)  # (num_objects, num_envs)
    active_index = torch.tensor([1, 0, 1], dtype=torch.long)
    env_arange = torch.arange(offsets.shape[1])

    active_offsets = offsets[active_index, env_arange]

    assert active_offsets.tolist() == [12, 4, 20]


def test_compact_gather_and_segmented_reductions_support_unequal_particle_counts():
    """The active view must gather only selected slice entries, not padded neighbours."""
    wp.init()
    source = wp.array(np.arange(27, dtype=np.float32).reshape(9, 3), dtype=wp.vec3f, device="cpu")
    # env 0 selects a two-particle slice at 1; env 1 selects a three-particle slice at 6.
    source_indices = wp.array(np.array([1, 2, 6, 7, 8], dtype=np.int32), device="cpu")
    compact = wp.empty(5, dtype=wp.vec3f, device="cpu")
    wp.launch(gather_compact_particles_vec3f, dim=5, inputs=[source, source_indices], outputs=[compact], device="cpu")

    values = torch.from_numpy(compact.numpy())
    env_ids = torch.tensor([0, 0, 1, 1, 1])
    assert torch.equal(values, torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]]))
    assert torch.equal(segmented_min(values, env_ids, 2), torch.tensor([[3.0, 4.0, 5.0], [18.0, 19.0, 20.0]]))
    assert torch.equal(segmented_max(values, env_ids, 2), torch.tensor([[6.0, 7.0, 8.0], [24.0, 25.0, 26.0]]))
