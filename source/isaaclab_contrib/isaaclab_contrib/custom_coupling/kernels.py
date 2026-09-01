# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernel used by the custom coupling manager."""

import warp as wp
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _eval_soft_ef_contact,
)


@wp.kernel
def _kernel_body_particle_reaction(
    contact_count: wp.array(dtype=wp.int32),
    contact_shape: wp.array(dtype=wp.int32),
    contact_indices: wp.array(dtype=wp.vec3i),
    contact_barycentric: wp.array(dtype=wp.vec3),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_q_prev: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=wp.int32),
    shape_margin: wp.array(dtype=wp.float32),
    soft_contact_ke: float,
    soft_contact_kd: float,
    soft_contact_mu: float,
    friction_epsilon: float,
    dt: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Apply body-particle contact reactions to rigid bodies.

    Newton's contact model evaluates normal, damping, and Coulomb friction forces on each particle. This kernel
    applies the equal and opposite force and torque to the contacted rigid body. One thread runs per allocated contact
    slot, and unused slots exit through ``contact_count``. ``particle_q_prev`` is SolverVBD's cached particle history,
    because VBD mutates ``particle_q`` in place.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    shape_idx = contact_shape[tid]
    if shape_idx < 0 or shape_idx >= shape_body.shape[0]:
        return
    body_idx = shape_body[shape_idx]
    if body_idx < 0:
        return

    corners = contact_indices[tid]
    if corners[0] < 0 or corners[0] >= particle_q.shape[0]:
        return

    # Newton's soft-contact buffer contains particle, edge, and face records.
    # Always use the same barycentric evaluator as VBD so the rigid reaction is
    # applied at the actual world-space contact point.
    bary = contact_barycentric[tid]
    particle_force, _hessian, contact_pos = _eval_soft_ef_contact(
        tid,
        corners,
        bary,
        particle_q,
        particle_q_prev,
        particle_radius,
        soft_contact_ke,
        soft_contact_kd,
        soft_contact_mu,
        friction_epsilon,
        shape_body,
        body_q,
        body_q_prev,
        body_qd,
        body_com,
        contact_shape,
        contact_body_pos,
        contact_body_vel,
        contact_normal,
        shape_margin,
        dt,
    )

    # Apply the equal and opposite particle force as a rigid-body wrench.
    com_pos = wp.transform_point(body_q[body_idx], body_com[body_idx])
    reaction = -particle_force
    torque = wp.cross(contact_pos - com_pos, reaction)
    wp.atomic_add(
        body_f,
        body_idx,
        wp.spatial_vector(
            reaction[0],
            reaction[1],
            reaction[2],
            torque[0],
            torque[1],
            torque[2],
        ),
    )
