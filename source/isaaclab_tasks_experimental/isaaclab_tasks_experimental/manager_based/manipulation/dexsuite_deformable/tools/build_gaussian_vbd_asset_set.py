# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a solver-compatible set of deformable assets from Gaussian splat USDs.

The generated assets have a common number of simulation vertices, a common
Y-up-to-Z-up / centred / scale convention, and a skinned
``ParticleField3DGaussianSplat`` visual.  They are the inputs to DexSuite's
``NewtonVbdTetAssetCfg`` and can be selected by a multi-asset spawner.

Pipeline per input asset
------------------------
1. Read the Gaussian positions from ``ParticleField3DGaussianSplat``.
2. Deterministically choose ``target_num_vertices`` positions with farthest
   point sampling; this creates a common particle budget.
3. Tetrahedralize the selected points with ``scipy.spatial.Delaunay``.
4. Write a legacy VBD tet source and package it with Gaussian-to-tet
   barycentric skinning in a standard ``UsdGeom.TetMesh`` USD.

With ``--collision-shell 0`` this deliberately uses a *coarse convex-hull
proxy* for the initial multi-asset workflow.  With a non-zero collision shell,
the tool instead reconstructs a solid voxel volume from the Gaussian radii,
erodes it by the contact radius, and tetrahedralizes samples from that inner
volume.  The visual Gaussians remain on the outer surface, so the VBD particle
contact shell reaches the visual object rather than sitting outside it.

Required offline dependencies: ``numpy``, ``scipy``, and the Isaac Sim/USD
Python environment (``pxr``).  SciPy is intentionally imported only when the
tetrahedralization step runs so task configuration imports stay dependency-free.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.spawners import (
    _surface_faces_from_tets,
)
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.tools.package_skinned_gaussian_tet_asset import (  # noqa: E501
    _load_gaussian_prim_data,
    package_skinned_gaussian_tet_asset,
)
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.tools.prepare_vbd_tet_assets import (
    write_vbd_tet_asset,
)


@dataclass(frozen=True)
class GaussianVbdAssetInfo:
    """Metadata needed to select and audit one generated solver asset."""

    source_gaussian_usd: str
    vbd_tet_usd: str
    packaged_usd: str
    num_vertices: int
    num_tets: int
    source_max_extent: float
    normalization_scale: float
    skinning_inside_fraction: float
    skinning_max_violation: float
    collision_shell_inset: float


def farthest_point_sample(points: np.ndarray, count: int, *, max_candidates: int = 50_000) -> np.ndarray:
    """Select a deterministic, spatially distributed subset of point positions.

    Candidate thinning is deterministic and keeps the cost bounded for large
    Gaussian fields.  The first point is farthest from the mean, which avoids
    a source-file-order-dependent seed.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Gaussian positions with shape (N, 3), got {points.shape}.")
    if count < 4:
        raise ValueError("target_num_vertices must be at least 4.")
    # Exact duplicate splats do not create distinct Newton particles and make
    # Delaunay cells singular. Sorting here also makes duplicate handling stable.
    points = np.unique(points, axis=0)
    if len(points) < count:
        raise ValueError(f"Need at least {count} unique Gaussian positions, got {len(points)}.")
    if max_candidates < count:
        raise ValueError("max_candidates must be at least target_num_vertices.")

    if len(points) > max_candidates:
        candidate_ids = np.linspace(0, len(points) - 1, max_candidates, dtype=np.int64)
        candidates = points[candidate_ids]
    else:
        candidates = points

    chosen = np.empty(count, dtype=np.int64)
    mean = candidates.mean(axis=0)
    min_distance_sq = np.sum((candidates - mean) ** 2, axis=1)
    chosen[0] = int(np.argmax(min_distance_sq))
    min_distance_sq = np.sum((candidates - candidates[chosen[0]]) ** 2, axis=1)
    for index in range(1, count):
        chosen[index] = int(np.argmax(min_distance_sq))
        distance_sq = np.sum((candidates - candidates[chosen[index]]) ** 2, axis=1)
        np.minimum(min_distance_sq, distance_sq, out=min_distance_sq)
    return np.ascontiguousarray(candidates[chosen], dtype=np.float32)


def tetrahedralize_proxy(vertices: np.ndarray, *, min_six_volume: float = 1.0e-10) -> np.ndarray:
    """Tetrahedralize a point proxy and discard numerically degenerate cells."""
    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise ImportError(
            "Gaussian VBD preparation needs scipy. Install scipy in the offline asset-preparation environment."
        ) from exc

    try:
        # QJ resolves co-spherical / nearly co-planar splat centres deterministically in Qhull.
        tets = np.asarray(Delaunay(vertices, qhull_options="Qbb Qc QJ").simplices, dtype=np.int32)
    except Exception as exc:
        raise ValueError("Delaunay tetrahedralization failed; the Gaussian positions may be near-coplanar.") from exc
    if not len(tets):
        raise ValueError("Delaunay tetrahedralization produced no tetrahedra.")

    tet_points = vertices[tets]
    signed_six_volume = np.einsum(
        "ij,ij->i",
        tet_points[:, 1] - tet_points[:, 0],
        np.cross(tet_points[:, 2] - tet_points[:, 0], tet_points[:, 3] - tet_points[:, 0]),
    )
    tolerance = float(min_six_volume) * float(np.ptp(vertices, axis=0).max()) ** 3
    keep = np.abs(signed_six_volume) > tolerance
    tets = tets[keep].copy()
    signed_six_volume = signed_six_volume[keep]
    if not len(tets):
        raise ValueError("All generated tetrahedra were degenerate.")
    negative = signed_six_volume < 0.0
    if np.any(negative):
        tets[negative, 2], tets[negative, 3] = tets[negative, 3].copy(), tets[negative, 2].copy()
    return np.ascontiguousarray(tets, dtype=np.int32)


def filter_spawner_degenerate_tets(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Remove cells rejected by the task spawner's absolute winding tolerance.

    Delaunay runs before normalization.  A cell with a tiny, nonzero source
    volume can become ``np.isclose(volume, 0)`` after conversion to the common
    20 cm task scale.  Apply exactly the same degeneracy predicate used by
    :func:`_ensure_positive_tet_winding` after scaling, before the VBD USD is
    written.
    """
    tet_points = vertices[tets]
    signed_six_volume = np.einsum(
        "ij,ij->i",
        tet_points[:, 1] - tet_points[:, 0],
        np.cross(tet_points[:, 2] - tet_points[:, 0], tet_points[:, 3] - tet_points[:, 0]),
    )
    keep = ~np.isclose(signed_six_volume, 0.0)
    result = tets[keep].copy()
    signed_six_volume = signed_six_volume[keep]
    if not len(result):
        raise ValueError("All tetrahedra become degenerate after task-scale normalization.")
    negative = signed_six_volume < 0.0
    if np.any(negative):
        result[negative, 2], result[negative, 3] = result[negative, 3].copy(), result[negative, 2].copy()
    return np.ascontiguousarray(result, dtype=np.int32)


def inset_tet_surface_for_collision_shell(
    vertices: np.ndarray,
    tets: np.ndarray,
    collision_shell: float,
) -> tuple[np.ndarray, float]:
    """Inset a tet proxy so vertex-centred collision spheres meet its visual surface.

    Newton contact acts on a particle sphere around every tet vertex.  When a
    surface vertex lies on the visual surface, the collision body is inflated
    by ``particle_radius``.  This routine moves surface vertices inward by the
    requested shell distance and uses backtracking to preserve positive tet
    volumes.  The Gaussian field is deliberately *not* moved; it is skinned
    to this inset proxy and remains the outer visual surface.
    """
    if collision_shell < 0.0:
        raise ValueError("collision_shell must be non-negative.")
    if collision_shell == 0.0:
        return np.ascontiguousarray(vertices, dtype=np.float32), 0.0

    surface_faces = _surface_faces_from_tets(tets)
    surface_ids = np.unique(surface_faces.reshape(-1))
    center = vertices.mean(axis=0)
    normals = np.zeros_like(vertices, dtype=np.float64)
    for face in surface_faces:
        points = vertices[face]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        if np.dot(normal, points.mean(axis=0) - center) < 0.0:
            normal = -normal
        normals[face] += normal
    lengths = np.linalg.norm(normals[surface_ids], axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("Could not determine outward normals for the tet proxy surface.")
    normals[surface_ids] /= lengths[:, None]

    requested = float(collision_shell)
    applied = requested
    min_six_volume = 1.0e-9 * float(np.ptp(vertices, axis=0).max()) ** 3
    while applied >= requested / 128.0:
        candidate = np.asarray(vertices, dtype=np.float32).copy()
        candidate[surface_ids] -= applied * normals[surface_ids].astype(np.float32)
        p = candidate[tets]
        signed_six_volume = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
        if np.all(signed_six_volume > min_six_volume):
            return np.ascontiguousarray(candidate), applied
        applied *= 0.5
    raise ValueError(
        f"Cannot inset this tet proxy by {requested:.6g} m without inverting tetrahedra; "
        "use a denser tetrahedral proxy or an SDF-eroded remeshing backend."
    )


def _gaussian_radii(gaussian_prim, expected_count: int) -> np.ndarray:
    """Return one conservative physical radius per Gaussian splat.

    ParticleField ``scales`` are metric axis radii for the inputs used by this
    task.  A sphere with the largest axis radius is conservative for a voxel
    solid and avoids relying on orientation support in an offline tool.
    """
    scales_attr = gaussian_prim.GetAttribute("scales")
    if not scales_attr.IsValid() or not scales_attr.HasValue():
        raise ValueError(f"Gaussian prim '{gaussian_prim.GetPath()}' has no scales attribute.")
    scales = np.asarray(scales_attr.Get(), dtype=np.float32)
    if scales.ndim != 2 or scales.shape[1] != 3 or len(scales) < expected_count:
        raise ValueError(
            f"Gaussian scales must have at least {expected_count} rows of float3 values, got {scales.shape}."
        )
    scales = scales[:expected_count]
    radii = np.max(scales, axis=1)
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("Gaussian scales must be finite and strictly positive.")
    return np.ascontiguousarray(radii, dtype=np.float32)


def _voxel_solid_from_gaussians(
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    voxel_size: float,
    collision_shell: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build an eroded dense occupancy grid from a Gaussian-splat field.

    This is a distance-field reconstruction: each splat contributes a sphere
    with its largest Gaussian axis radius.  We fill enclosed cavities before
    eroding so a dense surface-only Gaussian capture becomes a simulation
    volume.  The output grid contains centres that are at least
    ``collision_shell`` inside that reconstructed visual volume.
    """
    try:
        from scipy import ndimage
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("Collision-aware Gaussian VBD preparation needs scipy.") from exc

    if voxel_size <= 0.0:
        raise ValueError("collision_voxel_size must be positive.")
    if collision_shell <= 0.0:
        raise ValueError("collision_shell must be positive for voxel reconstruction.")
    if len(positions) != len(radii):
        raise ValueError("Gaussian positions and radii must have equal length.")

    bounds_min = np.min(positions - radii[:, None], axis=0) - voxel_size
    bounds_max = np.max(positions + radii[:, None], axis=0) + voxel_size
    shape = np.ceil((bounds_max - bounds_min) / voxel_size).astype(np.int64) + 1
    if np.any(shape > 192):
        raise ValueError(f"Collision grid shape {tuple(shape)} is too large; increase --collision-voxel-size.")

    axes = [bounds_min[axis] + voxel_size * np.arange(shape[axis], dtype=np.float32) for axis in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    tree = cKDTree(positions)
    max_radius = float(np.max(radii))
    occupancy = np.empty(len(grid), dtype=bool)
    for start in range(0, len(grid), 65_536):
        end = min(start + 65_536, len(grid))
        points = grid[start:end]
        # Nearest centre is not necessarily the splat that contains a point:
        # a large-radius Gaussian can sit behind many smaller, closer centres.
        # Query every centre that could overlap, then apply its own radius.
        candidates = tree.query_ball_point(points, r=max_radius, workers=-1)
        occupancy[start:end] = np.fromiter(
            (
                any(np.linalg.norm(point - positions[index]) <= radii[index] for index in indices)
                for point, indices in zip(points, candidates, strict=True)
            ),
            dtype=bool,
            count=len(points),
        )
    occupancy = occupancy.reshape(tuple(shape))
    # Close sub-voxel splat gaps, then turn the Gaussian shell into a solid.
    occupancy = ndimage.binary_closing(occupancy, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    occupancy = ndimage.binary_fill_holes(occupancy)
    # A distance threshold gives the requested physical shell directly.  The
    # previous integer-iteration erosion rounded every request up to whole
    # voxels (e.g. a 4 mm shell became nearly 7 mm on a 3.5 mm grid).
    distance_to_surface = ndimage.distance_transform_edt(occupancy, sampling=voxel_size)
    inner = distance_to_surface >= collision_shell
    if not np.any(inner):
        raise ValueError(
            "Eroding the reconstructed Gaussian volume by the collision shell removed the whole asset; "
            "use a smaller Newton particle radius or a larger normalized asset."
        )
    applied_shell = float(distance_to_surface[inner].min())
    return inner, bounds_min.astype(np.float32), float(voxel_size), applied_shell


def _sample_collision_aware_proxy(
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    target_num_vertices: int,
    collision_shell: float,
    voxel_size: float,
    allow_convex_fallback: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate a fixed-budget, non-convex tet proxy from an eroded voxel solid."""
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError("Collision-aware Gaussian VBD preparation needs scipy.") from exc

    inner, origin, spacing, applied_shell = _voxel_solid_from_gaussians(
        positions, radii, voxel_size=voxel_size, collision_shell=collision_shell
    )
    cell_indices = np.argwhere(inner)
    if len(cell_indices) < target_num_vertices:
        raise ValueError(
            f"The eroded collision volume has only {len(cell_indices)} voxel samples, fewer than the requested "
            f"{target_num_vertices} simulation vertices. Decrease --collision-voxel-size or target_num_vertices."
        )
    candidates = origin[None, :] + spacing * cell_indices.astype(np.float32)
    vertices = farthest_point_sample(candidates, target_num_vertices, max_candidates=max(50_000, target_num_vertices))
    all_tets = tetrahedralize_proxy(vertices)

    # Retain cells that lie inside (or immediately next to) the eroded solid.
    # Testing centroid and face-centroids rejects convex-hull bridges across
    # concavities while retaining a closed boundary at voxel precision.
    allowed = ndimage.binary_dilation(inner, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    tet_points = vertices[all_tets]
    samples = np.concatenate(
        (
            tet_points.mean(axis=1),
            tet_points[:, [0, 1, 2]].mean(axis=1),
            tet_points[:, [0, 1, 3]].mean(axis=1),
            tet_points[:, [0, 2, 3]].mean(axis=1),
            tet_points[:, [1, 2, 3]].mean(axis=1),
        ),
        axis=0,
    )
    voxel_indices = np.rint((samples - origin[None, :]) / spacing).astype(np.int64)
    in_bounds = np.all((voxel_indices >= 0) & (voxel_indices < np.asarray(allowed.shape)[None, :]), axis=1)
    is_allowed = np.zeros(len(samples), dtype=bool)
    is_allowed[in_bounds] = allowed[tuple(voxel_indices[in_bounds].T)]
    allowed_samples = is_allowed.reshape(5, len(all_tets))
    keep = allowed_samples.all(axis=0)
    tets = all_tets[keep]
    if not len(tets):
        raise ValueError("Collision-aware filtering removed every tetrahedron; use a finer collision voxel size.")
    used = np.unique(tets.reshape(-1))
    if len(used) != target_num_vertices:
        # Very thin appendages can have a valid tet centroid while one of its
        # face centroids crosses a one-voxel boundary.  Preserve their sampled
        # vertex rather than leaving an unconnected Newton particle; centroid
        # filtering still rejects the long convex-hull bridges that matter for
        # this reconstruction.
        tets = all_tets[allowed_samples[0]]
        used = np.unique(tets.reshape(-1))
    if len(used) != target_num_vertices:
        if not allow_convex_fallback:
            raise ValueError(
                "Collision-aware concavity filtering isolated simulation vertices. "
                "Use a larger shared vertex budget or explicitly pass --allow-convex-fallback."
            )
        warnings.warn(
            "Collision-aware concavity filtering isolated simulation vertices; "
            "using the explicitly requested full Delaunay fallback, which may bridge concavities.",
            stacklevel=2,
        )
        tets = all_tets
        used = np.unique(tets.reshape(-1))
    if len(used) != target_num_vertices:
        raise ValueError("Delaunay tetrahedralization did not connect every requested simulation vertex.")
    return np.ascontiguousarray(vertices, dtype=np.float32), np.ascontiguousarray(tets, dtype=np.int32), applied_shell


def _scaled_proxy_vertices(
    points: np.ndarray, vertices: np.ndarray, target_max_extent: float
) -> tuple[np.ndarray, float]:
    """Scale proxy vertices about the Gaussian mean while preserving skinning alignment."""
    if target_max_extent <= 0.0:
        raise ValueError("target_max_extent must be positive.")
    source_extent = float(np.ptp(points, axis=0).max())
    if source_extent <= 0.0:
        raise ValueError("Gaussian field has zero spatial extent.")
    scale = target_max_extent / source_extent
    gaussian_center = points.mean(axis=0, keepdims=True)
    # The package step centres Gaussian and tet fields independently. Matching their
    # pre-centering means makes those transforms identical and preserves skinning.
    proxy = (vertices - vertices.mean(axis=0, keepdims=True)) + gaussian_center
    proxy = (proxy - gaussian_center) * scale + gaussian_center
    return np.ascontiguousarray(proxy, dtype=np.float32), float(scale)


def build_gaussian_vbd_asset_set(
    gaussian_usd_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    target_num_vertices: int,
    target_max_extent: float = 0.20,
    collision_shell: float = 0.0,
    collision_voxel_size: float | None = None,
    skinning_chunk_size: int = 512,
    allow_convex_fallback: bool = False,
    append_manifest: bool = False,
    gaussian_prim_path: str | None = None,
    source_y_up: bool = True,
    max_candidates: int = 50_000,
    max_gaussians: int | None = None,
) -> list[GaussianVbdAssetInfo]:
    """Create uniformly-sized, skinned VBD assets from Gaussian-splat USD files.

    The output's *vertex count* is guaranteed to equal
    ``target_num_vertices`` for every asset. Tetrahedron counts are recorded
    but are intentionally not constrained: Newton's particle state is sized by
    vertices, while a tet connectivity can validly vary per asset.
    """
    paths = [Path(path) for path in gaussian_usd_paths]
    if not paths:
        raise ValueError("At least one Gaussian USD asset is required.")
    if skinning_chunk_size <= 0:
        raise ValueError("skinning_chunk_size must be positive.")

    output_dir = Path(output_dir)
    tet_dir = output_dir / "vbd_tets"
    packaged_dir = output_dir / "packaged"
    manifest_path = output_dir / "manifest.json"
    previous_entries: list[dict] = []
    if append_manifest and manifest_path.is_file():
        previous_entries = json.loads(manifest_path.read_text())
        if not isinstance(previous_entries, list):
            raise ValueError(f"Existing manifest '{manifest_path}' must contain a JSON list.")
    result: list[GaussianVbdAssetInfo] = []
    for source_path in paths:
        _, gaussian_prim, positions = _load_gaussian_prim_data(
            str(source_path), gaussian_prim_path, max_gaussians=max_gaussians
        )
        source_extent = float(np.ptp(positions, axis=0).max())
        if source_extent <= 0.0:
            raise ValueError("Gaussian field has zero spatial extent.")
        scale = target_max_extent / source_extent
        if collision_shell > 0.0:
            radii = _gaussian_radii(gaussian_prim, len(positions)) * scale
            center = positions.mean(axis=0, keepdims=True)
            normalized_positions = (positions - center) * scale + center
            if collision_voxel_size is None:
                # About 96 cells across the visual extent.  A coarser grid is
                # acceptable for a smaller contact shell: the eroded volume is
                # conservative by whole voxel cells, and keeping this bound
                # prevents a tiny radius from exploding the batch grid.
                voxel_size = max(target_max_extent / 96.0, collision_shell / 3.0)
            else:
                voxel_size = collision_voxel_size
            vertices, tets, applied_collision_shell = _sample_collision_aware_proxy(
                normalized_positions,
                radii,
                target_num_vertices=target_num_vertices,
                collision_shell=collision_shell,
                voxel_size=voxel_size,
                allow_convex_fallback=allow_convex_fallback,
            )
        else:
            vertices = farthest_point_sample(positions, target_num_vertices, max_candidates=max_candidates)
            tets = tetrahedralize_proxy(vertices)
            vertices, scale = _scaled_proxy_vertices(positions, vertices, target_max_extent)
            applied_collision_shell = 0.0
        tets = filter_spawner_degenerate_tets(vertices, tets)

        tet_path = tet_dir / f"{source_path.stem}_vbd_tet.usda"
        packaged_path = packaged_dir / f"{source_path.stem}_skinned_vbd_tet.usda"
        write_vbd_tet_asset(tet_path, vertices, tets)
        skinning = package_skinned_gaussian_tet_asset(
            gaussian_usd_path=str(source_path),
            tet_usd_path=str(tet_path),
            output_usd_path=str(packaged_path),
            gaussian_prim_path=gaussian_prim_path,
            rotate_tet_y_up_to_z_up=source_y_up,
            center_tet_to_origin=True,
            rotate_gaussian_y_up_to_z_up=source_y_up,
            center_gaussian_to_origin=True,
            gaussian_scale=(scale, scale, scale),
            chunk_size=skinning_chunk_size,
            max_gaussians=max_gaussians,
        )
        violations = skinning.barycentric_violation
        result.append(
            GaussianVbdAssetInfo(
                source_gaussian_usd=str(source_path),
                vbd_tet_usd=str(tet_path),
                packaged_usd=str(packaged_path),
                num_vertices=target_num_vertices,
                num_tets=int(len(tets)),
                source_max_extent=source_extent,
                normalization_scale=scale,
                skinning_inside_fraction=float(np.mean(violations <= 0.0)),
                skinning_max_violation=float(violations.max(initial=0.0)),
                collision_shell_inset=applied_collision_shell,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    new_entries = [asdict(asset) for asset in result]
    if append_manifest:
        new_sources = {entry["source_gaussian_usd"] for entry in new_entries}
        new_entries = [
            entry for entry in previous_entries if entry.get("source_gaussian_usd") not in new_sources
        ] + new_entries
    manifest_path.write_text(json.dumps(new_entries, indent=2) + "\n")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gaussian_usds", nargs="+", help="USDs containing ParticleField3DGaussianSplat prims.")
    parser.add_argument("--output-dir", required=True, help="Output root for VBD sources, packaged USDs, and manifest.")
    parser.add_argument("--target-num-vertices", required=True, type=int, help="Shared Newton particle budget.")
    parser.add_argument(
        "--target-max-extent", type=float, default=0.20, help="Shared longest rest-space extent in metres."
    )
    parser.add_argument(
        "--collision-shell",
        type=float,
        default=0.0,
        help="Inset physics tet surface in metres; use the configured Newton particle radius.",
    )
    parser.add_argument(
        "--collision-voxel-size",
        type=float,
        default=None,
        help="Voxel spacing in metres for collision-aware reconstruction (default derives from extent and shell).",
    )
    parser.add_argument("--gaussian-prim-path", default=None, help="Optional ParticleField3DGaussianSplat prim path.")
    parser.add_argument("--source-z-up", action="store_true", help="Inputs are already Z-up; skip Y-up conversion.")
    parser.add_argument("--max-candidates", type=int, default=50_000, help="FPS candidate cap for large splat fields.")
    parser.add_argument("--max-gaussians", type=int, default=None, help="Optional visual/debug splat cap.")
    parser.add_argument(
        "--skinning-chunk-size",
        type=int,
        default=512,
        help="Gaussians processed per barycentric-skinning chunk; lower values reduce preparation memory.",
    )
    parser.add_argument(
        "--append-manifest",
        action="store_true",
        help="Preserve entries for earlier one-asset invocations and replace only entries rebuilt by this command.",
    )
    parser.add_argument(
        "--allow-convex-fallback",
        action="store_true",
        help="Allow full Delaunay connectivity when strict collision-volume filtering isolates vertices.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    assets = build_gaussian_vbd_asset_set(
        args.gaussian_usds,
        args.output_dir,
        target_num_vertices=args.target_num_vertices,
        target_max_extent=args.target_max_extent,
        collision_shell=args.collision_shell,
        collision_voxel_size=args.collision_voxel_size,
        skinning_chunk_size=args.skinning_chunk_size,
        allow_convex_fallback=args.allow_convex_fallback,
        append_manifest=args.append_manifest,
        gaussian_prim_path=args.gaussian_prim_path,
        source_y_up=not args.source_z_up,
        max_candidates=args.max_candidates,
        max_gaussians=args.max_gaussians,
    )
    for asset in assets:
        print(
            f"built {asset.packaged_usd}: vertices={asset.num_vertices} tets={asset.num_tets} "
            f"inside={asset.skinning_inside_fraction:.3f}"
        )
