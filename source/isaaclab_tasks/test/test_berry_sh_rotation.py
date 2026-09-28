# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent basis and active-rotation checks for prepared Gaussian radiance."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.special import sph_harm_y

from isaaclab_tasks.contrib.berry_pick.sh_rotation import proper_rotation, rotate_sh, sh_basis


def test_live_renderer_guard_rejects_ghosting_version(monkeypatch):
    from isaaclab_tasks.contrib.berry_pick import render_settings

    monkeypatch.setattr(
        render_settings, "version", lambda name: {"ovrtx": "0.5.0.377615", "ovstage": "0.2.0.377349"}[name]
    )
    with pytest.raises(RuntimeError, match="ghosted"):
        render_settings.require_live_gaussian_renderer()
    monkeypatch.setattr(render_settings, "version", lambda name: {"ovrtx": "0.6.0", "ovstage": "0.3.0.0"}[name])
    render_settings.require_live_gaussian_renderer()
    monkeypatch.setattr(
        render_settings,
        "version",
        lambda name: {"ovrtx": "0.6.0.dev382408+mr50035.92a010ff", "ovstage": "0.3.0.382327"}[name],
    )
    render_settings.require_live_gaussian_renderer()


def reference_basis(directions):
    # SciPy's complex harmonics provide an independent implementation, including
    # Condon-Shortley signs. MDL/Graphdeco use imag for negative m, real for positive.
    theta = np.arccos(directions[..., 2])
    phi = np.arctan2(directions[..., 1], directions[..., 0])
    return np.stack(
        [
            sph_harm_y(degree, 0, theta, phi).real
            if m == 0
            else np.sqrt(2)
            * (sph_harm_y(degree, abs(m), theta, phi).imag if m < 0 else sph_harm_y(degree, m, theta, phi).real)
            for degree in range(4)
            for m in range(-degree, degree + 1)
        ],
        axis=-1,
    )


def test_sh_basis_matches_independent_complex_harmonics():
    d = np.random.default_rng(45).normal(size=(64, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    np.testing.assert_allclose(sh_basis(d), reference_basis(d), atol=2.0e-15)


def test_active_rotation_preserves_radiance_and_band_energy():
    rng = np.random.default_rng(17)
    coefficients = rng.normal(size=(8, 16, 3)).astype(np.float32)
    rotation = Rotation.random(8, random_state=rng).as_matrix()
    d = rng.normal(size=(43, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    transported = rotate_sh(coefficients, rotation)
    expected = np.einsum("ndk,nkc->ndc", reference_basis(np.einsum("dj,njk->ndk", d, rotation)), coefficients)
    actual = np.einsum("dk,nkc->ndc", reference_basis(d), transported)
    np.testing.assert_allclose(actual, expected, atol=3.0e-7)
    for degree in range(4):
        band = slice(degree * degree, (degree + 1) * (degree + 1))
        np.testing.assert_allclose(
            np.linalg.norm(transported[:, band], axis=1), np.linalg.norm(coefficients[:, band], axis=1), rtol=2.0e-7
        )
    np.testing.assert_allclose(rotate_sh(transported, rotation.transpose(0, 2, 1)), coefficients, atol=3.0e-7)


def test_settling_rotation_is_not_covariance_orientation_or_identity():
    rotation = Rotation.from_euler("xyz", [12, -59, 8], degrees=True).as_matrix()
    stretch = np.diag([0.7, 1.2, 0.95])
    recovered = proper_rotation((rotation @ stretch)[None])
    np.testing.assert_allclose(recovered[0], rotation, atol=1.0e-14)
    # A known l=1 signal radiates along +Z; after rotation it follows R @ +Z.
    coefficients = np.zeros((1, 16, 3), np.float64)
    coefficients[:, 2] = 1
    result = rotate_sh(coefficients, recovered)
    expected_axis = rotation[:, 2]
    np.testing.assert_allclose(
        result[0, 1:4, 0], [-expected_axis[1], expected_axis[2], -expected_axis[0]], atol=1.0e-14
    )
    assert not np.allclose(result, coefficients)  # The old settling bake omitted this step.


def test_runtime_quaternion_removes_rest_covariance_and_preserves_tint():
    import warp as wp

    from isaaclab_tasks.contrib.berry_pick._mpm.rtx_sh_frame import material_quaternion

    rng = np.random.default_rng(51)
    rotation = Rotation.random(8, random_state=rng).as_matrix()
    base = Rotation.random(8, random_state=rng).as_matrix() @ np.diag([0.0001, 0.0003, 0.0002])
    factors = rotation @ np.diag([0.7, 1.1, 0.9]) @ base
    damage = np.linspace(0, 1, 8).astype(np.float32)
    output = wp.empty(8, dtype=wp.vec4, device="cpu")
    wp.launch(
        material_quaternion,
        8,
        inputs=[
            wp.array(factors, dtype=wp.mat33, device="cpu"),
            wp.array(np.linalg.inv(base), dtype=wp.mat33, device="cpu"),
            wp.array(np.arange(8, dtype=np.int32), dtype=int, device="cpu"),
            wp.array(damage, dtype=float, device="cpu"),
            output,
        ],
        device="cpu",
    )
    packed = output.numpy()
    tint = np.linalg.norm(packed, axis=1)
    np.testing.assert_allclose(tint, 1 - 0.45 * damage, atol=2.0e-7)
    np.testing.assert_allclose(Rotation.from_quat(packed / tint[:, None]).as_matrix(), rotation, atol=1.0e-6)
