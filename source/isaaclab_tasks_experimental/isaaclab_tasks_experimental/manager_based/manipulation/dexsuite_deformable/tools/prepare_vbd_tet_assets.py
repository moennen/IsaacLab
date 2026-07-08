# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Normalize and validate VBD tet assets for heterogeneous DexSuite training.

This tool deliberately does *not* remesh an asset.  Remeshing a volume while
preserving a useful soft-body proxy needs a real tetrahedralization step (and
usually a watertight input surface) outside Isaac Lab.  Instead, it is the
last, deterministic preparation step after tetrahedralization:

* require a shared number of vertices and tetrahedra;
* apply the task's Y-up -> Z-up convention;
* centre the rest mesh at its vertex-mean COM proxy; and
* normalize its longest rest extent to a common value.

The output uses the ``vbd:vertices`` / ``vbd:tet_indices`` schema consumed by
``NewtonVbdTetAssetCfg``.  Therefore every output from one invocation has the
same tensor shape and can subsequently be passed to a multi-asset spawner.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VbdTetAssetInfo:
    """Topology and rest-space metadata emitted for a prepared tet asset."""

    path: str
    num_vertices: int
    num_tets: int
    rest_extent: tuple[float, float, float]
    normalization_scale: float


def _load_vbd_tet_asset(path: str | Path, source_prim_path: str = "/TetMesh") -> tuple[np.ndarray, np.ndarray]:
    """Read custom VBD vertices and tetrahedra without applying task transforms."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(f"Failed to open VBD tet asset: '{path}'.")
    prim = stage.GetPrimAtPath(source_prim_path)
    if not prim.IsValid():
        raise ValueError(f"Could not find '{source_prim_path}' in '{path}'.")

    vertices_attr = prim.GetAttribute("vbd:vertices")
    tets_attr = prim.GetAttribute("vbd:tet_indices")
    if not vertices_attr.IsValid() or not tets_attr.IsValid():
        raise ValueError(f"'{path}' must define vbd:vertices and vbd:tet_indices on '{source_prim_path}'.")

    vertices = np.asarray(vertices_attr.Get(), dtype=np.float32)
    indices = np.asarray(tets_attr.Get(), dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"'{path}' has invalid vbd:vertices shape {vertices.shape}; expected (V, 3).")
    if indices.ndim != 1 or indices.size % 4:
        raise ValueError(f"'{path}' has invalid vbd:tet_indices shape {indices.shape}; expected a flat 4*T array.")
    tets = indices.reshape(-1, 4).copy()
    if len(vertices) < 4 or len(tets) == 0:
        raise ValueError(f"'{path}' must contain at least four vertices and one tetrahedron.")
    if np.any(tets < 0) or np.any(tets >= len(vertices)):
        raise ValueError(f"'{path}' has tetrahedron indices outside [0, {len(vertices)}).")
    return np.ascontiguousarray(vertices), tets


def _ensure_positive_winding(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return tetrahedra with positive signed rest volume, rejecting degeneracy."""
    result = np.ascontiguousarray(tets, dtype=np.int32).copy()
    p = vertices[result]
    signed_six_volume = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    if np.any(np.isclose(signed_six_volume, 0.0)):
        raise ValueError("Tet mesh contains degenerate tetrahedra.")
    negative = signed_six_volume < 0.0
    if np.any(negative):
        result[negative, 2], result[negative, 3] = result[negative, 3].copy(), result[negative, 2].copy()
    return result


def normalize_vbd_tet_geometry(
    vertices: np.ndarray,
    tets: np.ndarray,
    *,
    source_y_up: bool = True,
    center_to_origin: bool = True,
    target_max_extent: float | None = 0.20,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply a shared axis, centre, scale, and winding convention to a tet mesh."""
    vertices = np.asarray(vertices, dtype=np.float32).copy()
    if source_y_up:
        vertices = np.stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]), axis=1)
    if center_to_origin:
        vertices -= vertices.mean(axis=0, keepdims=True)

    extent = np.ptp(vertices, axis=0)
    longest_extent = float(extent.max())
    if longest_extent <= 0.0:
        raise ValueError("Tet mesh has zero rest extent.")
    scale = 1.0
    if target_max_extent is not None:
        if target_max_extent <= 0.0:
            raise ValueError("target_max_extent must be positive when specified.")
        scale = float(target_max_extent) / longest_extent
        vertices *= scale

    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    return vertices, _ensure_positive_winding(vertices, tets), scale


def write_vbd_tet_asset(path: str | Path, vertices: np.ndarray, tets: np.ndarray) -> None:
    """Write a minimal VBD-compatible USD asset."""
    from pxr import Sdf, Usd

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    if stage is None:
        raise RuntimeError(f"Failed to create USD stage '{destination}'.")
    prim = stage.DefinePrim("/TetMesh", "Xform")
    prim.CreateAttribute("vbd:vertices", Sdf.ValueTypeNames.Point3fArray, custom=True).Set(vertices.tolist())
    prim.CreateAttribute("vbd:tet_indices", Sdf.ValueTypeNames.IntArray, custom=True).Set(tets.reshape(-1).tolist())
    stage.GetRootLayer().Save()


def prepare_vbd_tet_assets(
    asset_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    target_num_vertices: int,
    target_num_tets: int,
    source_y_up: bool = True,
    center_to_origin: bool = True,
    target_max_extent: float | None = 0.20,
) -> list[VbdTetAssetInfo]:
    """Prepare a topology-compatible asset set from already-tetrahedralized inputs.

    Counts are checked before writing anything.  This makes a failed batch a
    preparation error rather than a late Newton-model construction failure.
    """
    paths = [Path(path) for path in asset_paths]
    if not paths:
        raise ValueError("At least one tetrahedral asset is required.")
    if target_num_vertices < 4 or target_num_tets < 1:
        raise ValueError("target_num_vertices must be >= 4 and target_num_tets must be >= 1.")

    loaded = [(path, *_load_vbd_tet_asset(path)) for path in paths]
    mismatches = [
        f"{path}: {len(vertices)} vertices, {len(tets)} tetrahedra"
        for path, vertices, tets in loaded
        if len(vertices) != target_num_vertices or len(tets) != target_num_tets
    ]
    if mismatches:
        raise ValueError(
            "All inputs must already have the requested common topology budget; this tool does not remesh. "
            f"Expected {target_num_vertices} vertices and {target_num_tets} tetrahedra. Got: " + "; ".join(mismatches)
        )

    output_dir = Path(output_dir)
    result: list[VbdTetAssetInfo] = []
    for path, vertices, tets in loaded:
        vertices, tets, scale = normalize_vbd_tet_geometry(
            vertices,
            tets,
            source_y_up=source_y_up,
            center_to_origin=center_to_origin,
            target_max_extent=target_max_extent,
        )
        output_path = output_dir / f"{path.stem}_tet.usda"
        write_vbd_tet_asset(output_path, vertices, tets)
        extent = tuple(float(value) for value in np.ptp(vertices, axis=0))
        result.append(VbdTetAssetInfo(str(output_path), len(vertices), len(tets), extent, scale))

    (output_dir / "manifest.json").write_text(json.dumps([asdict(item) for item in result], indent=2) + "\n")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="+", help="Already tetrahedralized VBD USD assets.")
    parser.add_argument("--output-dir", required=True, help="Directory for normalized VBD USD assets and manifest.")
    parser.add_argument("--target-num-vertices", type=int, required=True)
    parser.add_argument("--target-num-tets", type=int, required=True)
    parser.add_argument("--target-max-extent", type=float, default=0.20)
    parser.add_argument("--source-z-up", action="store_true", help="Disable the default Y-up to Z-up conversion.")
    parser.add_argument("--keep-origin", action="store_true", help="Do not centre each rest mesh at its vertex mean.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepared = prepare_vbd_tet_assets(
        args.assets,
        args.output_dir,
        target_num_vertices=args.target_num_vertices,
        target_num_tets=args.target_num_tets,
        source_y_up=not args.source_z_up,
        center_to_origin=not args.keep_origin,
        target_max_extent=args.target_max_extent,
    )
    for asset in prepared:
        print(f"prepared {asset.path}: {asset.num_vertices} vertices, {asset.num_tets} tetrahedra")
