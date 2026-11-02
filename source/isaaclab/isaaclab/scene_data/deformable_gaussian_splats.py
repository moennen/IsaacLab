# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset-authored embeddings for Gaussian visuals driven by deformable nodes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp
from scipy import special
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from pxr import Usd

_CHANNEL_SETS = (
    ("centers",),
    ("centers", "covariance"),
    ("centers", "covariance", "sphericalHarmonics"),
)


@dataclass
class GaussianSplatsEmbedding:
    """One Gaussian prim's static, prototype-local embedding in simulation nodes.

    ``offsets`` delimit each splat's slice of ``indices`` and ``weights``. Node indices
    refer to the simulation geometry's point order, before clone-native offsets.
    Bind centers [m], orientations [xyzw], and scales [m] retain the asset's authored
    values; ``bind_transform`` maps Gaussian-local row vectors into the proxy parent frame.
    """

    path: str
    channels: tuple[str, ...]
    offsets: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    bind_centers: np.ndarray
    bind_orientations: np.ndarray
    bind_scales: np.ndarray
    sh_coefficients: np.ndarray
    sh_degree: int
    element_width: int
    bind_transform: np.ndarray
    normal_offsets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    """Signed bind distances from a surface proxy, preserved along its moving normals."""


def read_gaussian_splats_embedding(
    prim: Usd.Prim,
    sim_points: np.ndarray,
    sim_elements: np.ndarray,
    *,
    element_width: int,
    gaussian_splats_to_parent: np.ndarray | None = None,
) -> GaussianSplatsEmbedding:
    """Read an authored embedding or derive one from asset bind positions.

    The fallback selects a nearby element from centroid-tree candidates and
    computes barycentric coordinates for each center. Particle
    proxies have no elements and use the nearest simulation point.

    Args:
        prim: Gaussian visual prim with ``positions`` and optional Isaac Lab embedding attributes.
        sim_points: Simulation bind points in the Gaussian prim's parent frame.
        sim_elements: Flat proxy element connectivity, empty for particle proxies.
        element_width: Four for tetrahedra, three for triangles, zero for particles.
        gaussian_splats_to_parent: Row-vector affine transform from Gaussian local to parent frame.

    Returns:
        Validated embedding in simulation point order.
    """
    path = str(prim.GetPath())
    centers_value = prim.GetAttribute("positions").Get()
    if centers_value is None:
        raise ValueError(f"Gaussian prim {path!r} has no default positions.")
    centers = np.asarray(centers_value, dtype=np.float32).reshape(-1, 3)
    if gaussian_splats_to_parent is not None:
        centers = (centers @ gaussian_splats_to_parent[:3, :3] + gaussian_splats_to_parent[3, :3]).astype(np.float32)
    channels_value = prim.GetAttribute("isaaclab:deformableChannels").Get()
    channels = tuple(channels_value) if channels_value is not None else ("centers",)
    if channels not in _CHANNEL_SETS:
        raise ValueError(f"Invalid isaaclab:deformableChannels on {path!r}: {channels!r}.")

    names = ("isaaclab:embeddingOffsets", "isaaclab:embeddingIndices", "isaaclab:embeddingWeights")
    values = tuple(prim.GetAttribute(name).Get() for name in names)
    if any(value is not None for value in values):
        if any(value is None for value in values):
            raise ValueError(f"Gaussian splat prim {path!r} must author all three embedding arrays.")
        offsets = np.asarray(values[0], dtype=np.int32)
        indices = np.asarray(values[1], dtype=np.int32)
        weights = np.asarray(values[2], dtype=np.float32)
    elif element_width:
        elements = np.asarray(sim_elements, dtype=np.int32).reshape(-1, element_width)
        if not len(elements) and len(centers):
            raise ValueError(f"Gaussian prim {path!r} has no proxy elements for embedding.")
        offsets = np.arange(len(centers) + 1, dtype=np.int32) * element_width
        indices = np.empty((len(centers), element_width), dtype=np.int32)
        weights = np.empty((len(centers), element_width), dtype=np.float32)
        if len(centers):
            element_points = sim_points[elements]
            edges = (element_points[:, 1:] - element_points[:, :1]).transpose(0, 2, 1)
            valid = np.linalg.matrix_rank(edges) == element_width - 1
            if not np.any(valid):
                raise ValueError(f"Gaussian prim {path!r} has no nondegenerate proxy elements.")
            valid_elements = elements[valid]
            valid_points = element_points[valid]
            inverse_edges = np.linalg.pinv(edges[valid])
            tree = cKDTree(valid_points.mean(axis=1))
            candidate_count = min(32, len(valid_elements))
            for start in range(0, len(centers), 4096):
                chunk = centers[start : start + 4096]
                candidate_ids = np.asarray(tree.query(chunk, k=candidate_count)[1]).reshape(len(chunk), -1)
                coordinates = np.einsum(
                    "nkij,nkj->nki",
                    inverse_edges[candidate_ids],
                    chunk[:, None, :] - valid_points[candidate_ids, 0],
                )
                candidate_weights = np.concatenate(
                    (1.0 - coordinates.sum(axis=-1, keepdims=True), coordinates), axis=-1
                )
                score = np.maximum(-candidate_weights, 0.0).sum(axis=-1)
                selected = np.argmin(score, axis=1)
                rows = np.arange(len(chunk))
                indices[start : start + len(chunk)] = valid_elements[candidate_ids[rows, selected]]
                weights[start : start + len(chunk)] = candidate_weights[rows, selected]
        indices = indices.reshape(-1)
        weights = weights.reshape(-1)
    else:
        if not len(sim_points) and len(centers):
            raise ValueError(f"Gaussian prim {path!r} has no proxy points for embedding.")
        offsets = np.arange(len(centers) + 1, dtype=np.int32)
        indices = (
            np.asarray(cKDTree(sim_points).query(centers)[1], dtype=np.int32).reshape(-1)
            if len(centers)
            else np.empty(0, dtype=np.int32)
        )
        weights = np.ones(len(centers), dtype=np.float32)

    if len(offsets) != len(centers) + 1 or offsets[0] != 0 or offsets[-1] != len(indices):
        raise ValueError(f"Invalid Gaussian embedding offsets on {path!r}.")
    if np.any(np.diff(offsets) <= 0) or len(indices) != len(weights):
        raise ValueError(f"Invalid Gaussian embedding ranges on {path!r}.")
    if np.any(indices < 0) or np.any(indices >= len(sim_points)):
        raise ValueError(f"Gaussian embedding index is outside proxy points on {path!r}.")
    sums = np.add.reduceat(weights, offsets[:-1]) if len(centers) else np.empty(0)
    if not np.all(np.isfinite(weights)) or not np.allclose(sums, 1.0, atol=1.0e-4):
        raise ValueError(f"Gaussian embedding weights must be finite and sum to one on {path!r}.")
    if "covariance" in channels:
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            if element_width == 3 and end - start != 3:
                raise ValueError(f"Surface Gaussian covariance must bind three proxy nodes on {path!r}.")
            nodes = sim_points[indices[start:end]]
            rank = np.linalg.matrix_rank(nodes[1:] - nodes[0]) if len(nodes) > 1 else 0
            required_rank = 2 if element_width == 3 else 3
            if rank < required_rank:
                raise ValueError(f"Gaussian covariance embedding has degenerate proxy nodes on {path!r}.")
    count = len(centers)
    orientations = prim.GetAttribute("orientations").Get()
    scales = prim.GetAttribute("scales").Get()
    if orientations is None:
        orientations_array = np.tile([0.0, 0.0, 0.0, 1.0], (count, 1)).astype(np.float32)
    else:
        orientations_array = np.asarray(
            [[*value.GetImaginary(), value.GetReal()] for value in orientations], dtype=np.float32
        )
    scales_array = np.ones((count, 3), dtype=np.float32) if scales is None else np.asarray(scales, dtype=np.float32)
    degree = int(prim.GetAttribute("radiance:sphericalHarmonicsDegree").Get() or 0)
    coefficients = prim.GetAttribute("radiance:sphericalHarmonicsCoefficients").Get()
    coefficients_array = (
        np.empty((0, 3), dtype=np.float32) if coefficients is None else np.asarray(coefficients, dtype=np.float32)
    )
    if orientations_array.shape != (count, 4) or scales_array.shape != (count, 3):
        raise ValueError(f"Gaussian covariance arrays have the wrong size on {path!r}.")
    if (coefficients is not None or "sphericalHarmonics" in channels) and coefficients_array.shape != (
        count * (degree + 1) ** 2,
        3,
    ):
        raise ValueError(f"Gaussian spherical harmonics coefficients have the wrong size on {path!r}.")
    if (degree < 0 or degree > 3) and coefficients is not None:
        raise ValueError(f"Gaussian SH rotation supports degrees up to 3 on {path!r}.")
    normal_offsets = np.zeros(count, dtype=np.float32)
    if element_width == 3:
        for splat, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            if end - start != 3:
                continue
            nodes = sim_points[indices[start:end]]
            normal = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
            length = np.linalg.norm(normal)
            if length > 1.0e-12:
                center = np.einsum("n,nj->j", weights[start:end], nodes)
                normal_offsets[splat] = np.dot(centers[splat] - center, normal / length)
    return GaussianSplatsEmbedding(
        path,
        channels,
        offsets,
        indices,
        weights,
        centers,
        orientations_array,
        scales_array,
        coefficients_array,
        degree,
        element_width,
        np.eye(4, dtype=np.float32)
        if gaussian_splats_to_parent is None
        else gaussian_splats_to_parent.astype(np.float32),
        normal_offsets,
    )


@dataclass
class GaussianSplatsState:
    """World-space Gaussian arrays for renderer attribute writes.

    The fields are one-dimensional Warp arrays of vec3f, quatf, vec3f, and
    vec3f, respectively. SH coefficients have N * B entries, where B is the
    number of basis functions. Only SH may be ``None`` when it is not authored.
    """

    positions: wp.array
    orientations: wp.array
    scales: wp.array
    sh_coefficients: wp.array | None


def _sh_basis(directions: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate the real 3DGS spherical-harmonic basis through degree three."""
    x, y, z = directions.T
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    basis = []
    for level in range(degree + 1):
        for order in range(-level, level + 1):
            if hasattr(special, "sph_harm_y"):
                harmonic = special.sph_harm_y(level, abs(order), theta, phi)
            else:
                harmonic = special.sph_harm(abs(order), level, phi, theta)
            if order < 0:
                basis.append(np.sqrt(2.0) * harmonic.imag)
            elif order > 0:
                basis.append(np.sqrt(2.0) * harmonic.real)
            else:
                basis.append(harmonic.real)
    return np.column_stack(basis)


def _rotate_sh_many(coefficients: np.ndarray, rotations: np.ndarray, degree: int) -> np.ndarray:
    """Rotate batches of 3DGS coefficients with one least-squares solve per chunk."""
    if degree == 0:
        return coefficients.copy()
    samples = 2 * (degree + 1) ** 2
    index = np.arange(samples, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / samples
    phi = np.pi * (3.0 - np.sqrt(5.0)) * index
    radius = np.sqrt(1.0 - z * z)
    directions = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    target = _sh_basis(directions, degree)
    result = np.empty_like(coefficients)
    for start in range(0, len(coefficients), 4096):
        chunk = coefficients[start : start + 4096]
        rotated = np.matmul(directions[None], rotations[start : start + len(chunk)])
        source = _sh_basis(rotated.reshape(-1, 3), degree).reshape(len(chunk), samples, -1)
        values = np.einsum("nsb,nbc->nsc", source, chunk)
        rhs = values.transpose(1, 0, 2).reshape(samples, -1)
        solved = np.linalg.lstsq(target, rhs, rcond=None)[0]
        result[start : start + len(chunk)] = solved.reshape(-1, len(chunk), 3).transpose(1, 0, 2)
    result[:, 0] = coefficients[:, 0]
    return result


def _covariance_axes(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert batched covariance matrices to quaternion orientations and linear scales."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvectors[np.linalg.det(eigenvectors) < 0, :, 0] *= -1
    return Rotation.from_matrix(eigenvectors).as_quat().astype(np.float32), np.sqrt(
        np.maximum(eigenvalues, 0.0)
    ).astype(np.float32)


def _prepare_bind_frame(
    embedding: GaussianSplatsEmbedding,
    bind_proxy_points: np.ndarray,
    root_pos: tuple[float, float, float],
    root_rot: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the static world bind frame and Gaussian covariance once per asset."""
    root_matrix = Rotation.from_quat(root_rot).as_matrix()
    rest = np.asarray(bind_proxy_points, dtype=np.float64) @ root_matrix.T + np.asarray(root_pos)
    transform = root_matrix @ embedding.bind_transform[:3, :3].T
    left, _, right = np.linalg.svd(transform)
    bind_rotation = left @ right
    if np.linalg.det(bind_rotation) < 0:
        left[:, -1] *= -1
        bind_rotation = left @ right
    frames = transform @ Rotation.from_quat(embedding.bind_orientations).as_matrix()
    scaled_frames = frames * embedding.bind_scales[:, None, :]
    covariance = scaled_frames @ scaled_frames.transpose(0, 2, 1)
    return rest, transform, bind_rotation, covariance


@wp.func
def _gaussian_covariance_axes(covariance: wp.mat33f) -> tuple[wp.quatf, wp.vec3f]:
    """Diagonalize a covariance into a proper orientation and nonnegative scales."""
    axes, eigenvalues = wp.eig3(covariance)
    if wp.determinant(axes) < 0.0:
        axes[0, 0] = -axes[0, 0]
        axes[1, 0] = -axes[1, 0]
        axes[2, 0] = -axes[2, 0]
    scales = wp.vec3f(
        wp.sqrt(wp.max(eigenvalues[0], 0.0)),
        wp.sqrt(wp.max(eigenvalues[1], 0.0)),
        wp.sqrt(wp.max(eigenvalues[2], 0.0)),
    )
    return wp.quat_from_matrix(axes), scales


@wp.kernel(enable_backward=False)
def _evaluate_gaussian_splats_kernel(
    current: wp.array(dtype=wp.vec3f),
    rest: wp.array(dtype=wp.vec3f),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float32),
    normal_offsets: wp.array(dtype=wp.float32),
    rest_inverse: wp.array(dtype=wp.mat33f),
    bind_covariance: wp.array(dtype=wp.mat33f),
    bind_rotation: wp.array(dtype=wp.mat33f),
    element_width: int,
    deform_covariance: bool,
    deform_sh: bool,
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    scales: wp.array(dtype=wp.vec3f),
    rotations: wp.array(dtype=wp.mat33f),
):
    """Evaluate each asset embedding independently from native proxy nodes."""
    splat = wp.tid()
    start = offsets[splat]
    end = offsets[splat + 1]
    center = wp.vec3f(0.0)
    for slot in range(start, end):
        center += weights[slot] * current[indices[slot]]
    if element_width == 3 and end - start == 3:
        a = current[indices[start]]
        b = current[indices[start + 1]]
        c = current[indices[start + 2]]
        normal = wp.cross(b - a, c - a)
        if wp.length(normal) > 1.0e-12:
            center += normal_offsets[splat] * wp.normalize(normal)
    positions[splat] = center
    if not deform_covariance:
        return

    deformation = wp.mat33f(0.0)
    if element_width == 3:
        a = current[indices[start]]
        b = current[indices[start + 1]]
        c = current[indices[start + 2]]
        normal = wp.cross(b - a, c - a)
        if wp.length(normal) > 1.0e-12:
            normal = wp.normalize(normal)
        else:
            normal = wp.vec3f(0.0)
        edges = wp.mat33f(
            b[0] - a[0],
            c[0] - a[0],
            normal[0],
            b[1] - a[1],
            c[1] - a[1],
            normal[1],
            b[2] - a[2],
            c[2] - a[2],
            normal[2],
        )
        deformation = edges @ rest_inverse[splat]
    else:
        rest_center = wp.vec3f(0.0)
        for slot in range(start, end):
            rest_center += weights[slot] * rest[indices[slot]]
        for slot in range(start, end):
            node = indices[slot]
            deformation += wp.outer(current[node] - center, rest[node] - rest_center)
        deformation = deformation @ rest_inverse[splat]
    covariance = deformation @ bind_covariance[splat] @ wp.transpose(deformation)
    orientation, scale = _gaussian_covariance_axes(covariance)
    orientations[splat] = orientation
    scales[splat] = scale
    if deform_sh:
        left, _, right = wp.svd3(deformation)
        rotation = left @ wp.transpose(right)
        if wp.determinant(rotation) < 0.0:
            left[0, 2] = -left[0, 2]
            left[1, 2] = -left[1, 2]
            left[2, 2] = -left[2, 2]
            rotation = left @ wp.transpose(right)
        rotations[splat] = rotation @ bind_rotation[0]


@wp.func
def _gaussian_sh_basis_value(
    direction: wp.vec3f,
    basis: int,
    offsets: wp.array(dtype=wp.int32),
    terms: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.float32),
    exponents: wp.array(dtype=wp.vec3i),
) -> float:
    """Evaluate one sparse Cartesian polynomial representing a real SH basis function."""
    result = float(0.0)
    for slot in range(offsets[basis], offsets[basis + 1]):
        power = exponents[terms[slot]]
        monomial = float(1.0)
        for _ in range(power[0]):
            monomial *= direction[0]
        for _ in range(power[1]):
            monomial *= direction[1]
        for _ in range(power[2]):
            monomial *= direction[2]
        result += values[slot] * monomial
    return result


@wp.kernel(enable_backward=False)
def _sample_rotated_gaussian_sh_kernel(
    coefficients: wp.array(dtype=wp.vec3f),
    rotations: wp.array(dtype=wp.mat33f),
    directions: wp.array(dtype=wp.vec3f),
    basis_offsets: wp.array(dtype=wp.int32),
    basis_terms: wp.array(dtype=wp.int32),
    basis_values: wp.array(dtype=wp.float32),
    exponents: wp.array(dtype=wp.vec3i),
    basis_count: int,
    samples: wp.array(dtype=wp.vec3f),
):
    """Sample rotated appearance at the precomputed collocation directions."""
    splat, sample = wp.tid()
    direction = wp.transpose(rotations[splat]) @ directions[sample]
    value = wp.vec3f(0.0)
    for basis in range(basis_count):
        weight = _gaussian_sh_basis_value(direction, basis, basis_offsets, basis_terms, basis_values, exponents)
        value += weight * coefficients[splat * basis_count + basis]
    samples[splat * directions.shape[0] + sample] = value


@wp.kernel(enable_backward=False)
def _solve_rotated_gaussian_sh_kernel(
    coefficients: wp.array(dtype=wp.vec3f),
    samples: wp.array(dtype=wp.vec3f),
    inverse_basis: wp.array2d(dtype=wp.float32),
    basis_count: int,
    output: wp.array(dtype=wp.vec3f),
):
    """Recover rotated coefficients while preserving the constant SH term exactly."""
    splat, basis = wp.tid()
    if basis == 0:
        output[splat * basis_count] = coefficients[splat * basis_count]
        return
    value = wp.vec3f(0.0)
    for sample in range(inverse_basis.shape[1]):
        value += inverse_basis[basis, sample] * samples[splat * inverse_basis.shape[1] + sample]
    output[splat * basis_count + basis] = value


@wp.kernel(enable_backward=False)
def _gaussian_splats_to_local_kernel(
    shape_poses: wp.array(dtype=wp.transformf),
    shape_scales: wp.array(dtype=wp.vec3f),
    shape_index: int,
    world_positions: wp.array(dtype=wp.vec3f),
    world_orientations: wp.array(dtype=wp.quatf),
    world_scales: wp.array(dtype=wp.vec3f),
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    scales: wp.array(dtype=wp.vec3f),
    sh_rotations: wp.array(dtype=wp.mat33f),
):
    """Convert Gaussian world arrays to a Newton shape's local coordinates."""
    splat = wp.tid()
    pose = shape_poses[shape_index]
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    shape_scale = shape_scales[shape_index]
    inverse = wp.mat33f(
        1.0 / shape_scale[0],
        0.0,
        0.0,
        0.0,
        1.0 / shape_scale[1],
        0.0,
        0.0,
        0.0,
        1.0 / shape_scale[2],
    ) @ wp.transpose(rotation)
    positions[splat] = inverse @ (world_positions[splat] - wp.transform_get_translation(pose))
    frame = wp.quat_to_matrix(world_orientations[splat])
    scale = world_scales[splat]
    scaled = frame @ wp.mat33f(
        scale[0],
        0.0,
        0.0,
        0.0,
        scale[1],
        0.0,
        0.0,
        0.0,
        scale[2],
    )
    local_frame = inverse @ scaled
    covariance = local_frame @ wp.transpose(local_frame)
    orientation, local_scale = _gaussian_covariance_axes(covariance)
    orientations[splat] = orientation
    scales[splat] = local_scale
    if sh_rotations.shape[0] > 0:
        sh_rotations[splat] = wp.transpose(rotation)


@dataclass
class WarpGaussianSplatsEmbedding:
    """Asset embedding tables uploaded once to the proxy's Warp device."""

    embedding: GaussianSplatsEmbedding
    device: wp.Device
    offsets: wp.array
    indices: wp.array
    weights: wp.array
    normal_offsets: wp.array
    rest: wp.array
    rest_inverse: wp.array
    covariance: wp.array
    orientations: wp.array
    scales: wp.array
    coefficients: wp.array | None
    static_coefficients: wp.array | None
    bind_rotation: wp.array
    directions: wp.array | None
    inverse_basis: wp.array | None
    basis_offsets: wp.array | None
    basis_terms: wp.array | None
    basis_values: wp.array | None
    exponents: wp.array | None

    @classmethod
    def create(
        cls,
        embedding: GaussianSplatsEmbedding,
        bind_points: np.ndarray,
        root_pos: tuple[float, float, float],
        root_rot: tuple[float, float, float, float],
        device: wp.Device | str,
    ) -> WarpGaussianSplatsEmbedding:
        """Prepare geometry and SH fit tables before the first simulation step."""
        device = wp.get_device(device)
        rest, _, bind_rotation, covariance = _prepare_bind_frame(embedding, bind_points, root_pos, root_rot)
        orientations, scales = _covariance_axes(covariance)
        count = len(embedding.bind_centers)
        rest_inverse = np.zeros((count, 3, 3), dtype=np.float32)
        if "covariance" in embedding.channels:
            for splat, (start, end) in enumerate(zip(embedding.offsets[:-1], embedding.offsets[1:], strict=True)):
                nodes = rest[embedding.indices[start:end]]
                if embedding.element_width == 3:
                    edges = (nodes[1:] - nodes[:1]).T
                    normal = np.cross(edges[:, 0], edges[:, 1])
                    normal /= max(np.linalg.norm(normal), 1.0e-12)
                    rest_inverse[splat] = np.linalg.pinv(np.column_stack((edges, normal)))
                else:
                    center = np.einsum("n,nj->j", embedding.weights[start:end], nodes)
                    centered = nodes - center
                    rest_inverse[splat] = np.linalg.pinv(centered.T @ centered)
        basis_count = (embedding.sh_degree + 1) ** 2
        coefficients = static_coefficients = None
        directions = inverse_basis = basis_offsets = basis_terms = basis_values = exponents = None
        if len(embedding.sh_coefficients):
            source = embedding.sh_coefficients.reshape(count, basis_count, 3)
            coefficients = wp.array(embedding.sh_coefficients, dtype=wp.vec3f, device=device)
            if "sphericalHarmonics" not in embedding.channels or embedding.sh_degree == 0:
                baked = _rotate_sh_many(source, np.broadcast_to(bind_rotation, (count, 3, 3)), embedding.sh_degree)
                static_coefficients = wp.array(baked.reshape(-1, 3), dtype=wp.vec3f, device=device)
            if embedding.sh_degree > 0:
                sample_count = 2 * basis_count
                sample_index = np.arange(sample_count, dtype=np.float64) + 0.5
                z = 1.0 - 2.0 * sample_index / sample_count
                phi = np.pi * (3.0 - np.sqrt(5.0)) * sample_index
                radius = np.sqrt(1.0 - z * z)
                sample_directions = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
                fit_count = 256
                fit_index = np.arange(fit_count, dtype=np.float64) + 0.5
                fit_z = 1.0 - 2.0 * fit_index / fit_count
                fit_phi = np.pi * (3.0 - np.sqrt(5.0)) * fit_index
                fit_radius = np.sqrt(1.0 - fit_z * fit_z)
                fit_directions = np.column_stack((fit_radius * np.cos(fit_phi), fit_radius * np.sin(fit_phi), fit_z))
                powers = np.array(
                    [
                        (i, j, level - i - j)
                        for level in range(embedding.sh_degree + 1)
                        for i in range(level + 1)
                        for j in range(level - i + 1)
                    ],
                    dtype=np.int32,
                )
                monomials = np.column_stack([np.prod(fit_directions**power, axis=1) for power in powers])
                polynomial = np.linalg.lstsq(monomials, _sh_basis(fit_directions, embedding.sh_degree), rcond=None)[0]
                offsets = [0]
                terms = []
                values = []
                for basis in range(basis_count):
                    selected = np.flatnonzero(np.abs(polynomial[:, basis]) > 1.0e-6)
                    terms.extend(selected.tolist())
                    values.extend(polynomial[selected, basis].tolist())
                    offsets.append(len(terms))
                directions = wp.array(sample_directions.astype(np.float32), dtype=wp.vec3f, device=device)
                inverse_basis = wp.array(
                    np.linalg.pinv(_sh_basis(sample_directions, embedding.sh_degree)).astype(np.float32),
                    dtype=wp.float32,
                    device=device,
                )
                basis_offsets = wp.array(np.asarray(offsets, dtype=np.int32), device=device)
                basis_terms = wp.array(np.asarray(terms, dtype=np.int32), device=device)
                basis_values = wp.array(np.asarray(values, dtype=np.float32), device=device)
                exponents = wp.array(powers, dtype=wp.vec3i, device=device)
        return cls(
            embedding,
            device,
            wp.array(embedding.offsets, dtype=wp.int32, device=device),
            wp.array(embedding.indices, dtype=wp.int32, device=device),
            wp.array(embedding.weights, dtype=wp.float32, device=device),
            wp.array(embedding.normal_offsets, dtype=wp.float32, device=device),
            wp.array(rest.astype(np.float32), dtype=wp.vec3f, device=device),
            wp.array(rest_inverse, dtype=wp.mat33f, device=device),
            wp.array(covariance.astype(np.float32), dtype=wp.mat33f, device=device),
            wp.array(orientations, dtype=wp.quatf, device=device),
            wp.array(scales, dtype=wp.vec3f, device=device),
            coefficients,
            static_coefficients,
            wp.array(bind_rotation[None].astype(np.float32), dtype=wp.mat33f, device=device),
            directions,
            inverse_basis,
            basis_offsets,
            basis_terms,
            basis_values,
            exponents,
        )

    def _rotate_sh(self, coefficients: wp.array, rotations: wp.array) -> wp.array:
        """Rotate SH coefficients using device-resident collocation tables."""
        count = len(rotations)
        basis_count = (self.embedding.sh_degree + 1) ** 2
        samples = wp.empty(count * len(self.directions), dtype=wp.vec3f, device=self.device)
        output = wp.empty(count * basis_count, dtype=wp.vec3f, device=self.device)
        wp.launch(
            _sample_rotated_gaussian_sh_kernel,
            dim=(count, len(self.directions)),
            inputs=[
                coefficients,
                rotations,
                self.directions,
                self.basis_offsets,
                self.basis_terms,
                self.basis_values,
                self.exponents,
                basis_count,
                samples,
            ],
            device=self.device,
        )
        wp.launch(
            _solve_rotated_gaussian_sh_kernel,
            dim=(count, basis_count),
            inputs=[coefficients, samples, self.inverse_basis, basis_count, output],
            device=self.device,
        )
        return output

    def evaluate(self, current_points: wp.array) -> GaussianSplatsState:
        """Evaluate all dynamic channels on the proxy's Warp device."""
        count = len(self.embedding.bind_centers)
        positions = wp.empty(count, dtype=wp.vec3f, device=self.device)
        deform_covariance = "covariance" in self.embedding.channels
        deform_sh = "sphericalHarmonics" in self.embedding.channels and self.embedding.sh_degree > 0
        orientations = wp.empty(count, dtype=wp.quatf, device=self.device) if deform_covariance else self.orientations
        scales = wp.empty(count, dtype=wp.vec3f, device=self.device) if deform_covariance else self.scales
        rotations = wp.empty(count if deform_sh else 0, dtype=wp.mat33f, device=self.device)
        if count:
            wp.launch(
                _evaluate_gaussian_splats_kernel,
                dim=count,
                inputs=[
                    current_points,
                    self.rest,
                    self.offsets,
                    self.indices,
                    self.weights,
                    self.normal_offsets,
                    self.rest_inverse,
                    self.covariance,
                    self.bind_rotation,
                    self.embedding.element_width,
                    deform_covariance,
                    deform_sh,
                    positions,
                    orientations,
                    scales,
                    rotations,
                ],
                device=self.device,
            )
        coefficients = self.static_coefficients
        if deform_sh and count:
            coefficients = self._rotate_sh(self.coefficients, rotations)
        return GaussianSplatsState(positions, orientations, scales, coefficients)

    def to_local(
        self, state: GaussianSplatsState, shape_poses: wp.array, shape_scales: wp.array, shape_index: int
    ) -> GaussianSplatsState:
        """Convert world Gaussian arrays to a Newton shape's local frame on either Warp device."""
        count = len(state.positions)
        positions = wp.empty(count, dtype=wp.vec3f, device=self.device)
        orientations = wp.empty(count, dtype=wp.quatf, device=self.device)
        scales = wp.empty(count, dtype=wp.vec3f, device=self.device)
        rotate_sh = state.sh_coefficients is not None and self.embedding.sh_degree > 0
        rotations = wp.empty(count if rotate_sh else 0, dtype=wp.mat33f, device=self.device)
        if count:
            wp.launch(
                _gaussian_splats_to_local_kernel,
                dim=count,
                inputs=[
                    shape_poses,
                    shape_scales,
                    shape_index,
                    state.positions,
                    state.orientations,
                    state.scales,
                    positions,
                    orientations,
                    scales,
                    rotations,
                ],
                device=self.device,
            )
        coefficients = (
            self._rotate_sh(state.sh_coefficients, rotations) if rotate_sh and count else state.sh_coefficients
        )
        return GaussianSplatsState(positions, orientations, scales, coefficients)
