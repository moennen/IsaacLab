# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package a Gaussian visual asset with a tetrahedral physics proxy and skinning data.

The generated USD keeps the physics proxy as a standard ``UsdGeom.TetMesh`` and the
visual asset as ``ParticleField3DGaussianSplat``. Since no standardized USD schema
currently describes "Gaussian splats skinned to a deformable tet mesh", the binding
is stored in a custom Newton namespace on the Gaussian prim:

* ``newton:deformableSkin:targetTetMesh`` relationship to the TetMesh prim.
* ``newton:deformableSkin:influenceIndices`` flat int array, four tet-vertex ids per Gaussian.
* ``newton:deformableSkin:influenceWeights`` flat float array, four barycentric weights per Gaussian.
* ``newton:deformableSkin:tetIndices`` diagnostic tet id per Gaussian.

This intentionally mirrors USD mesh-skinning semantics without claiming UsdSkel
compatibility: tet vertices are the deformation handles, not skeleton joints.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.spawners import (
    _vbd_tet_asset_geometry,
)

DEFAULT_ROOT_PATH = "/SkinnedGaussianTetAsset"
DEFAULT_TET_PATH = f"{DEFAULT_ROOT_PATH}/physics/sim_tet_mesh"
DEFAULT_GAUSSIAN_PATH = f"{DEFAULT_ROOT_PATH}/visual/gaussians_0"


@dataclass(frozen=True)
class SkinningResult:
    """Barycentric Gaussian-to-tet embedding."""

    tet_indices: np.ndarray
    """Containing or nearest tet id per Gaussian, shape ``(N,)``."""

    influence_indices: np.ndarray
    """Tet vertex ids per Gaussian, shape ``(N, 4)``."""

    influence_weights: np.ndarray
    """Barycentric weights per Gaussian, shape ``(N, 4)``."""

    barycentric_violation: np.ndarray
    """Outside-tet violation metric per Gaussian. Zero means inside."""

    rest_tet_basis_inv: np.ndarray
    """Inverse rest basis per tet, shape ``(T, 3, 3)``."""


def _find_first_prim_by_type(stage: Usd.Stage, type_name: str) -> Usd.Prim:
    for prim in stage.Traverse():
        if prim.GetTypeName() == type_name:
            return prim
    raise ValueError(f"Could not find a '{type_name}' prim in stage '{stage.GetRootLayer().identifier}'.")


def _as_float3_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected an array with shape (N, 3), got {array.shape}.")
    return np.ascontiguousarray(array)


def _rotate_y_up_to_z_up_positions(positions: np.ndarray) -> np.ndarray:
    """Apply Y-up to Z-up coordinate conversion: ``(x, y, z) -> (x, -z, y)``."""
    return np.ascontiguousarray(np.stack((positions[:, 0], -positions[:, 2], positions[:, 1]), axis=1))


def _copy_attr(dst_prim: Usd.Prim, src_attr: Usd.Attribute) -> None:
    value = src_attr.Get()
    if value is None:
        return
    dst_attr = dst_prim.CreateAttribute(src_attr.GetName(), src_attr.GetTypeName(), custom=src_attr.IsCustom())
    dst_attr.Set(value)


def _load_gaussian_prim_data(
    gaussian_usd_path: str,
    gaussian_prim_path: str | None = None,
    *,
    rotate_y_up_to_z_up: bool = False,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center_to_origin: bool = False,
    max_gaussians: int | None = None,
) -> tuple[Usd.Stage, Usd.Prim, np.ndarray]:
    """Load and transform Gaussian rest positions used for skinning."""
    stage = Usd.Stage.Open(str(gaussian_usd_path))
    if stage is None:
        raise FileNotFoundError(f"Failed to open Gaussian USD: '{gaussian_usd_path}'.")
    gaussian_prim = (
        stage.GetPrimAtPath(gaussian_prim_path)
        if gaussian_prim_path
        else _find_first_prim_by_type(stage, "ParticleField3DGaussianSplat")
    )
    if not gaussian_prim.IsValid():
        raise ValueError(f"Could not find Gaussian prim '{gaussian_prim_path}' in '{gaussian_usd_path}'.")

    positions_attr = gaussian_prim.GetAttribute("positions")
    if not positions_attr.IsValid() or not positions_attr.HasValue():
        raise ValueError(f"Gaussian prim '{gaussian_prim.GetPath()}' has no positions attribute.")

    positions = _as_float3_array(positions_attr.Get())
    if max_gaussians is not None:
        if max_gaussians <= 0:
            raise ValueError("max_gaussians must be positive when provided.")
        positions = positions[:max_gaussians]

    if rotate_y_up_to_z_up:
        positions = _rotate_y_up_to_z_up_positions(positions)
    positions = positions * np.asarray(scale, dtype=np.float32).reshape(1, 3)
    if center_to_origin:
        positions = positions - positions.mean(axis=0, keepdims=True)

    return stage, gaussian_prim, np.ascontiguousarray(positions, dtype=np.float32)


def compute_tet_barycentric_skinning(
    query_points: np.ndarray,
    tet_vertices: np.ndarray,
    tets: np.ndarray,
    *,
    chunk_size: int = 4096,
    inside_tolerance: float = 1.0e-5,
) -> SkinningResult:
    """Compute four-vertex barycentric skinning weights from query points to a tet mesh.

    Points inside the tet volume bind to the containing tet. Points outside bind to
    the tet with the smallest barycentric violation and keep the unclamped weights,
    which gives a linear extrapolation useful for surface Gaussians slightly outside
    the coarse physics proxy.
    """
    points = np.ascontiguousarray(query_points, dtype=np.float32)
    vertices = np.ascontiguousarray(tet_vertices, dtype=np.float32)
    tet_indices = np.ascontiguousarray(tets, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected query_points shape (N, 3), got {points.shape}.")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected tet_vertices shape (V, 3), got {vertices.shape}.")
    if tet_indices.ndim != 2 or tet_indices.shape[1] != 4:
        raise ValueError(f"Expected tets shape (T, 4), got {tet_indices.shape}.")

    tet_points = vertices[tet_indices]
    basis = np.stack(
        (
            tet_points[:, 1] - tet_points[:, 0],
            tet_points[:, 2] - tet_points[:, 0],
            tet_points[:, 3] - tet_points[:, 0],
        ),
        axis=-1,
    )
    try:
        rest_tet_basis_inv = np.linalg.inv(basis).astype(np.float32)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Tet mesh contains a degenerate tetrahedron.") from exc

    selected_tets = np.empty(points.shape[0], dtype=np.int32)
    selected_weights = np.empty((points.shape[0], 4), dtype=np.float32)
    selected_violation = np.empty(points.shape[0], dtype=np.float32)

    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        rel = points[start:end, None, :] - tet_points[None, :, 0, :]
        uvw = np.einsum("tij,btj->bti", rest_tet_basis_inv, rel, optimize=True)
        weights = np.empty((end - start, tet_indices.shape[0], 4), dtype=np.float32)
        weights[..., 0] = 1.0 - uvw.sum(axis=-1)
        weights[..., 1:] = uvw

        min_weight = weights.min(axis=-1)
        # Zero for inside points, positive for outside/extrapolated points.
        violation = np.maximum(-min_weight - inside_tolerance, 0.0)
        best_tet = np.argmin(violation, axis=1).astype(np.int32)
        rows = np.arange(end - start)
        selected_tets[start:end] = best_tet
        selected_weights[start:end] = weights[rows, best_tet]
        selected_violation[start:end] = violation[rows, best_tet]

    return SkinningResult(
        tet_indices=selected_tets,
        influence_indices=tet_indices[selected_tets],
        influence_weights=selected_weights,
        barycentric_violation=selected_violation,
        rest_tet_basis_inv=rest_tet_basis_inv,
    )


def reconstruct_points_from_skinning(tet_vertices: np.ndarray, result: SkinningResult) -> np.ndarray:
    """Reconstruct skinned rest positions from tet vertices and skinning weights."""
    return np.einsum(
        "nij,ni->nj",
        np.asarray(tet_vertices, dtype=np.float32)[result.influence_indices],
        result.influence_weights,
        optimize=True,
    )


def _write_tet_mesh(
    stage: Usd.Stage,
    tet_path: str,
    vertices: np.ndarray,
    tets: np.ndarray,
    surface_faces: np.ndarray,
) -> Usd.Prim:
    parent_path = str(Sdf.Path(tet_path).GetParentPath())
    if parent_path and parent_path != "/":
        stage.DefinePrim(parent_path, "Scope")
    tet = UsdGeom.TetMesh.Define(stage, tet_path)
    tet.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices])
    tet.CreateTetVertexIndicesAttr([Gf.Vec4i(int(a), int(b), int(c), int(d)) for a, b, c, d in tets])
    tet.CreateSurfaceFaceVertexIndicesAttr([Gf.Vec3i(int(a), int(b), int(c)) for a, b, c in surface_faces])
    return tet.GetPrim()


def _write_gaussian_prim(
    stage: Usd.Stage,
    gaussian_path: str,
    source_gaussian_prim: Usd.Prim,
    positions: np.ndarray,
    *,
    max_gaussians: int | None = None,
) -> Usd.Prim:
    parent_path = str(Sdf.Path(gaussian_path).GetParentPath())
    if parent_path and parent_path != "/":
        stage.DefinePrim(parent_path, "Scope")
    gaussian_prim = stage.DefinePrim(gaussian_path, "ParticleField3DGaussianSplat")
    source_positions_count = len(source_gaussian_prim.GetAttribute("positions").Get())
    copied_positions = False
    for attr in source_gaussian_prim.GetAttributes():
        if attr.GetName() == "positions":
            gaussian_prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set(
                [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in positions]
            )
            copied_positions = True
            continue
        value = attr.Get()
        if max_gaussians is not None and hasattr(value, "__len__"):
            value_len = len(value)
            if value_len == source_positions_count:
                value = value[:max_gaussians]
            elif source_positions_count > 0 and value_len % source_positions_count == 0:
                value = value[: max_gaussians * (value_len // source_positions_count)]
        if value is not None:
            dst_attr = gaussian_prim.CreateAttribute(attr.GetName(), attr.GetTypeName(), custom=attr.IsCustom())
            dst_attr.Set(value)

    if not copied_positions:
        gaussian_prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set(
            [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in positions]
        )
    return gaussian_prim


def _write_skinning_attrs(
    gaussian_prim: Usd.Prim,
    tet_prim: Usd.Prim,
    result: SkinningResult,
    *,
    source_gaussian_usd: str,
    source_tet_usd: str,
) -> None:
    n = int(result.influence_indices.shape[0])
    gaussian_prim.CreateRelationship("newton:deformableSkin:targetTetMesh").SetTargets([tet_prim.GetPath()])
    gaussian_prim.CreateAttribute("newton:deformableSkin:schemaVersion", Sdf.ValueTypeNames.Int).Set(1)
    gaussian_prim.CreateAttribute("newton:deformableSkin:method", Sdf.ValueTypeNames.Token).Set("tetBarycentricLinear")
    gaussian_prim.CreateAttribute("newton:deformableSkin:influenceSize", Sdf.ValueTypeNames.Int).Set(4)
    gaussian_prim.CreateAttribute("newton:deformableSkin:pointCount", Sdf.ValueTypeNames.Int).Set(n)
    gaussian_prim.CreateAttribute("newton:deformableSkin:influenceIndices", Sdf.ValueTypeNames.IntArray).Set(
        [int(v) for v in result.influence_indices.reshape(-1)]
    )
    gaussian_prim.CreateAttribute("newton:deformableSkin:influenceWeights", Sdf.ValueTypeNames.FloatArray).Set(
        [float(v) for v in result.influence_weights.reshape(-1)]
    )
    gaussian_prim.CreateAttribute("newton:deformableSkin:tetIndices", Sdf.ValueTypeNames.IntArray).Set(
        [int(v) for v in result.tet_indices]
    )
    gaussian_prim.CreateAttribute("newton:deformableSkin:barycentricViolation", Sdf.ValueTypeNames.FloatArray).Set(
        [float(v) for v in result.barycentric_violation]
    )
    gaussian_prim.CreateAttribute("newton:deformableSkin:sourceGaussianUsd", Sdf.ValueTypeNames.String).Set(
        source_gaussian_usd
    )
    gaussian_prim.CreateAttribute("newton:deformableSkin:sourceTetUsd", Sdf.ValueTypeNames.String).Set(source_tet_usd)

    tet_prim.CreateAttribute("newton:deformableSkin:restTetBasisInv", Sdf.ValueTypeNames.FloatArray).Set(
        [float(v) for v in result.rest_tet_basis_inv.reshape(-1)]
    )
    tet_prim.CreateAttribute("newton:deformableSkin:restTetBasisInvLayout", Sdf.ValueTypeNames.Token).Set("rowMajor3x3")


def package_skinned_gaussian_tet_asset(
    *,
    gaussian_usd_path: str,
    tet_usd_path: str,
    output_usd_path: str,
    gaussian_prim_path: str | None = None,
    tet_source_prim_path: str = "/TetMesh",
    root_path: str = DEFAULT_ROOT_PATH,
    tet_path: str | None = None,
    gaussian_path: str | None = None,
    rotate_tet_y_up_to_z_up: bool = True,
    center_tet_to_origin: bool = True,
    rotate_gaussian_y_up_to_z_up: bool = False,
    center_gaussian_to_origin: bool = False,
    gaussian_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    chunk_size: int = 4096,
    inside_tolerance: float = 1.0e-5,
    max_gaussians: int | None = None,
) -> SkinningResult:
    """Create a combined Gaussian + TetMesh USD with custom deformable skinning data."""
    source_gaussian_stage, source_gaussian_prim, gaussian_positions = _load_gaussian_prim_data(
        gaussian_usd_path,
        gaussian_prim_path,
        rotate_y_up_to_z_up=rotate_gaussian_y_up_to_z_up,
        scale=gaussian_scale,
        center_to_origin=center_gaussian_to_origin,
        max_gaussians=max_gaussians,
    )
    # Keep the source stage alive while copying attributes from the source prim.
    _ = source_gaussian_stage
    tet_vertices, tets, surface_faces = _vbd_tet_asset_geometry(
        tet_usd_path,
        source_prim_path=tet_source_prim_path,
        rotate_y_up_to_z_up=rotate_tet_y_up_to_z_up,
        center_to_origin=center_tet_to_origin,
    )
    result = compute_tet_barycentric_skinning(
        gaussian_positions,
        tet_vertices,
        tets,
        chunk_size=chunk_size,
        inside_tolerance=inside_tolerance,
    )

    output_path = Path(output_usd_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    root_prim = stage.DefinePrim(root_path, "Xform")
    stage.SetDefaultPrim(root_prim)

    tet_prim = _write_tet_mesh(
        stage, tet_path or f"{root_path}/physics/sim_tet_mesh", tet_vertices, tets, surface_faces
    )
    gaussian_prim = _write_gaussian_prim(
        stage,
        gaussian_path or f"{root_path}/visual/gaussians_0",
        source_gaussian_prim,
        gaussian_positions,
        max_gaussians=max_gaussians,
    )
    _write_skinning_attrs(
        gaussian_prim,
        tet_prim,
        result,
        source_gaussian_usd=str(gaussian_usd_path),
        source_tet_usd=str(tet_usd_path),
    )
    stage.Save()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gaussian-usd", required=True, help="Source USD containing ParticleField3DGaussianSplat.")
    parser.add_argument("--tet-usd", required=True, help="Source legacy VBD tet USD containing vbd:* attributes.")
    parser.add_argument("--output", required=True, help="Output USD/USDC path.")
    parser.add_argument("--gaussian-prim-path", default=None, help="Optional source Gaussian prim path.")
    parser.add_argument("--tet-source-prim-path", default="/TetMesh", help="Source tet prim path.")
    parser.add_argument("--root-path", default=DEFAULT_ROOT_PATH, help="Output root prim path.")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Gaussian chunk size for skinning.")
    parser.add_argument("--inside-tolerance", type=float, default=1.0e-5, help="Barycentric inside tolerance.")
    parser.add_argument(
        "--max-gaussians", type=int, default=None, help="Debug option: package only the first N splats."
    )
    parser.add_argument(
        "--gaussian-y-up",
        action="store_true",
        help="Rotate Gaussian positions from Y-up to Z-up before packaging.",
    )
    parser.add_argument(
        "--center-gaussian",
        action="store_true",
        help="Subtract the Gaussian mean before skinning and writing positions.",
    )
    parser.add_argument(
        "--no-center-tet",
        action="store_true",
        help="Do not recenter the tet vertices. The task default is centered.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = package_skinned_gaussian_tet_asset(
        gaussian_usd_path=args.gaussian_usd,
        tet_usd_path=args.tet_usd,
        output_usd_path=args.output,
        gaussian_prim_path=args.gaussian_prim_path,
        tet_source_prim_path=args.tet_source_prim_path,
        root_path=args.root_path,
        center_tet_to_origin=not args.no_center_tet,
        rotate_gaussian_y_up_to_z_up=args.gaussian_y_up,
        center_gaussian_to_origin=args.center_gaussian,
        chunk_size=args.chunk_size,
        inside_tolerance=args.inside_tolerance,
        max_gaussians=args.max_gaussians,
    )
    inside_fraction = float(np.mean(result.barycentric_violation <= 0.0)) if result.tet_indices.size else 0.0
    print(
        "packaged skinned gaussian tet asset: "
        f"points={result.tet_indices.size} "
        f"inside_fraction={inside_fraction:.4f} "
        f"max_violation={float(result.barycentric_violation.max(initial=0.0)):.6g}"
    )


if __name__ == "__main__":
    main()
