# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical state and reset contract of the optional Kaolin tissue adapter."""

from itertools import product

import newton
import numpy as np
import pytest
import warp as wp

from isaaclab_tasks.contrib.franka_pick_berries.physics.grasp_simplicits import (
    GraspSimplicitsSolverCfg,
    SolverSimplicits,
    _body_transforms,
)

pytestmark = pytest.mark.kitless


@pytest.mark.skipif(not wp.is_cuda_available(), reason="Kaolin capture requires CUDA")
@pytest.mark.parametrize("captured", [False, True])
def test_simplicits_falls_under_newton_gravity_and_replays_after_reset(captured):
    """Potential-gradient sign, published particle velocity, and reset must agree in eager and captured execution."""
    pytest.importorskip("kaolin")
    device = "cuda:0"
    points = np.asarray(list(product((-0.01, 0.01), repeat=3)), np.float32)
    points[:, 2] += 0.12
    count = len(points)
    data = {
        "pts": points,
        "yms": np.full(count, 12000, np.float32),
        "prs": np.full(count, 0.4, np.float32),
        "rhos": np.full(count, 1000, np.float32),
        "appx_vol": np.asarray(8e-6, np.float32),
        "skinning_weights": np.ones((count, 1), np.float32),
        "dwdx": np.zeros((count, 1, 3), np.float32),
    }
    normals = points - points.mean(0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    data["collision"] = {
        "pts": points,
        "weights": data["skinning_weights"],
        "dwdx": data["dwdx"],
        "normals": normals,
        "areas": np.full(count, 0.0024 / count, np.float32),
    }
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
    builder.add_body(
        xform=wp.transform(wp.vec3(0.2, 0, 0.1), wp.quat_identity()), is_kinematic=True, label="panda_leftfinger"
    )
    builder.add_body(
        xform=wp.transform(wp.vec3(-0.2, 0, 0.1), wp.quat_identity()), is_kinematic=True, label="panda_rightfinger"
    )
    for point in points:
        builder.add_particle(pos=wp.vec3(*point), vel=wp.vec3(0), mass=0.008 / count, radius=0.001)
    model = builder.finalize(device=device)
    cfg = GraspSimplicitsSolverCfg(
        representations=[data],
        placements=[np.eye(4, dtype=np.float32)],
        captured=captured,
        contact_points=32,
        timestep=0.0025,
    )
    solver = SolverSimplicits(model, cfg)
    state = model.state()
    dt = 0.01

    def rollout():
        # The backend warms up once, then captures the complete coupled step.
        solver.step(state, state, model.control(), None, dt)
        snapshots = [state.particle_q.numpy().copy()]
        if captured and solver.supports_outer_capture:
            with wp.ScopedCapture(device=device) as capture:
                solver.step(state, state, model.control(), None, dt)
        for _ in range(2):
            if captured and solver.supports_outer_capture:
                wp.capture_launch(capture.graph)
            else:
                solver.step(state, state, model.control(), None, dt)
            snapshots.append(state.particle_q.numpy().copy())
        return snapshots

    snapshots = rollout()
    center = snapshots[-1].mean(0)
    assert center[2] < points[:, 2].mean() - 0.002
    np.testing.assert_allclose(center[:2], [0, 0], atol=1e-5)
    np.testing.assert_allclose(state.particle_qd.numpy().mean(0), [0, 0, -9.81 * 3 * dt], atol=0.02)
    wp.copy(state.particle_q, model.particle_q)
    state.particle_qd.zero_()
    solver.reset(state)
    for actual, expected in zip(rollout(), snapshots, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_pad_motion_uses_newton_linear_then_angular_twist():
    """Translation and rotation must follow Newton's documented velocity convention."""
    output = wp.empty(2, dtype=wp.mat44f, device="cpu")
    wp.launch(
        _body_transforms,
        2,
        [
            wp.array([wp.transform_identity()] * 2, dtype=wp.transformf, device="cpu"),
            wp.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, np.pi]], dtype=wp.spatial_vector, device="cpu"),
            wp.zeros(2, dtype=wp.vec3f, device="cpu"),
            wp.array([0, 1], dtype=int, device="cpu"),
            0.5,
            output,
        ],
        device="cpu",
    )
    expected = np.repeat(np.eye(4)[None], 2, axis=0)
    expected[0, 0, 3] = 0.5
    expected[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    np.testing.assert_allclose(output.numpy(), expected, atol=1e-6)


def test_surface_is_outward_for_mixed_tet_winding_and_rejects_degenerate_tets():
    from isaaclab_tasks.contrib.franka_pick_berries.scripts.prepare_simplicits_asset import _surface

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    faces = _surface(vertices, np.array([[0, 1, 2, 3], [0, 1, 2, 4]]))
    assert len(faces) == 6
    points = vertices[faces]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    assert np.all(np.sum(normals * (points.mean(1) - vertices.mean(0)), axis=1) > 0)
    with pytest.raises(ValueError, match="degenerate"):
        _surface(vertices, np.array([[0, 1, 2, 2]]))


def test_missing_gaussian_weights_has_actionable_error():
    from pxr import Sdf, Usd

    from isaaclab_tasks.contrib.franka_pick_berries.assets.simplicits import COLLISION_PATH, load_simplicits

    stage = Usd.Stage.CreateInMemory()
    field = stage.DefinePrim("/Berry/Gaussians")
    field.CreateAttribute("kaolin_skinned_physics:default:pts", Sdf.ValueTypeNames.Point3fArray).Set([(0, 0, 0)])
    stage.DefinePrim(COLLISION_PATH)
    with pytest.raises(ValueError, match="Gaussian skinning weights"):
        load_simplicits(stage)
