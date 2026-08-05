# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a custom VBD tetrahedral-mesh USD to a standalone standard USD asset.

The source must contain one prim with ``vbd:vertices`` and
``vbd:tet_indices``.  The output has ``/Asset`` as its default prim, a
``UsdGeom.TetMesh`` at ``/Asset/SimulationMesh``, and its derived triangle
boundary at ``/Asset/VisualMesh``.

Example:

    uv run python scripts/tools/convert_vbd_tetmesh_usd.py \\
        --input-usd /path/to/source.usda --output-usd /path/to/vbd_asset.usda
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

parser = argparse.ArgumentParser(description="Convert a custom VBD tetmesh USD into a standard simulation asset.")
parser.add_argument(
    "--input-usd", type=Path, required=True, help="USD/USDZ containing vbd:vertices and vbd:tet_indices."
)
parser.add_argument("--output-usd", type=Path, required=True, help="Destination USDA/USD file; it will be overwritten.")
parser.add_argument(
    "--tetmesh-prim-path",
    type=str,
    default=None,
    help="Source prim path. Required only when the input contains more than one VBD tetmesh payload.",
)
args_cli = parser.parse_args()

if not args_cli.input_usd.is_file():
    parser.error(f"--input-usd does not exist or is not a file: {args_cli.input_usd}")
if args_cli.output_usd.parent and not args_cli.output_usd.parent.is_dir():
    parser.error(f"Parent directory does not exist: {args_cli.output_usd.parent}")
if args_cli.output_usd.resolve() == args_cli.input_usd.resolve():
    parser.error("--output-usd must differ from --input-usd.")


def _as_tets(indices) -> list[tuple[int, int, int, int]]:
    """Convert a flattened VBD index array into one four-tuple per tetrahedron."""
    flattened = [int(index) for index in indices]
    if len(flattened) % 4:
        raise ValueError(f"vbd:tet_indices must have a multiple of four entries; got {len(flattened)}.")
    return [tuple(flattened[index : index + 4]) for index in range(0, len(flattened), 4)]


def _normalize_tet_winding(points, tets: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Return positively oriented tetrahedra, rejecting degenerate elements."""
    normalized_tets = []
    for tet_index, (a, b, c, d) in enumerate(tets):
        pa, pb, pc, pd = (points[index] for index in (a, b, c, d))
        ab = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        ac = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
        ad = (pd[0] - pa[0], pd[1] - pa[1], pd[2] - pa[2])
        signed_six_volume = (
            ab[0] * (ac[1] * ad[2] - ac[2] * ad[1])
            - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
            + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
        )
        if signed_six_volume == 0.0:
            raise ValueError(f"VBD tetmesh contains degenerate tetrahedron {tet_index}: {a}, {b}, {c}, {d}.")
        normalized_tets.append((a, c, b, d) if signed_six_volume < 0.0 else (a, b, c, d))
    return normalized_tets


def _surface_faces(tets: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int]]:
    """Return only faces incident to exactly one tetrahedron."""
    face_counts: dict[tuple[int, int, int], int] = {}
    face_winding: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for a, b, c, d in tets:
        for face in ((a, c, b), (b, c, d), (a, b, d), (a, d, c)):
            key = tuple(sorted(face))
            face_counts[key] = face_counts.get(key, 0) + 1
            face_winding.setdefault(key, face)
    return [face_winding[key] for key, count in face_counts.items() if count == 1]


def main() -> None:
    source_path = args_cli.input_usd.resolve()
    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise ValueError(f"Unable to open USD stage: {source_path}")

    candidates = [
        prim
        for prim in source_stage.Traverse()
        if prim.GetAttribute("vbd:vertices").IsValid() and prim.GetAttribute("vbd:tet_indices").IsValid()
    ]
    if args_cli.tetmesh_prim_path:
        source_prim = source_stage.GetPrimAtPath(args_cli.tetmesh_prim_path)
        if source_prim not in candidates:
            raise ValueError(f"--tetmesh-prim-path is not a VBD tetmesh payload: {args_cli.tetmesh_prim_path}")
    elif len(candidates) == 1:
        source_prim = candidates[0]
    elif not candidates:
        raise ValueError(f"No prim with vbd:vertices and vbd:tet_indices found in {source_path}")
    else:
        paths = ", ".join(prim.GetPath().pathString for prim in candidates)
        raise ValueError(f"Found multiple VBD tetmesh payloads; pass --tetmesh-prim-path. Candidates: {paths}")

    points = source_prim.GetAttribute("vbd:vertices").Get()
    tets = _as_tets(source_prim.GetAttribute("vbd:tet_indices").Get())
    if not points or not tets:
        raise ValueError(f"VBD tetmesh '{source_prim.GetPath()}' has no vertices or tetrahedra.")
    if min(index for tet in tets for index in tet) < 0 or max(index for tet in tets for index in tet) >= len(points):
        raise ValueError(f"VBD tetmesh '{source_prim.GetPath()}' has indices outside its {len(points)} vertices.")
    tets = _normalize_tet_winding(points, tets)

    output_path = args_cli.output_usd.resolve()
    output_stage = Usd.Stage.CreateNew(str(output_path))
    asset = UsdGeom.Xform.Define(output_stage, "/Asset")
    output_stage.SetDefaultPrim(asset.GetPrim())

    simulation = UsdGeom.TetMesh.Define(output_stage, "/Asset/SimulationMesh")
    simulation.GetPointsAttr().Set(points)
    simulation.GetTetVertexIndicesAttr().Set([Gf.Vec4i(*tet) for tet in tets])

    visual = UsdGeom.Mesh.Define(output_stage, "/Asset/VisualMesh")
    visual.GetPointsAttr().Set(points)
    surface_faces = _surface_faces(tets)
    visual.GetFaceVertexCountsAttr().Set([3] * len(surface_faces))
    visual.GetFaceVertexIndicesAttr().Set([index for face in surface_faces for index in face])
    visual.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    output_stage.GetRootLayer().Save()
    print(
        f"[INFO] Wrote {output_path}: {len(points)} vertices, {len(tets)} tetrahedra, "
        f"{len(surface_faces)} surface faces."
    )


if __name__ == "__main__":
    main()
