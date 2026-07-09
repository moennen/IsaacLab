# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export ``ParticleField3DGaussianSplat`` USD prims to 3D Gaussian-splat PLY.

The output uses the property layout consumed by VoMP's ``Gaussian.load_ply``.
It deliberately has no VoMP dependency, so it can run in the Isaac Sim asset
preparation environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pxr import Usd


def export_gaussian_usd_to_ply(
    usd_path: str | Path,
    output_path: str | Path,
    gaussian_prim_path: str | None = None,
    *,
    target_max_extent: float | None = 0.20,
    source_y_up: bool = True,
) -> None:
    """Export one Gaussian USD asset to a standard binary little-endian PLY.

    Args:
        usd_path: Source USD containing a ``ParticleField3DGaussianSplat`` prim.
        output_path: Destination PLY path.
        gaussian_prim_path: Optional prim path. The first Gaussian prim is used when omitted.
    """
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"Failed to open Gaussian USD: '{usd_path}'.")
    prim = (
        stage.GetPrimAtPath(gaussian_prim_path)
        if gaussian_prim_path
        else next(
            (candidate for candidate in stage.Traverse() if candidate.GetTypeName() == "ParticleField3DGaussianSplat"),
            None,
        )
    )
    if prim is None or not prim.IsValid():
        raise ValueError(f"Invalid Gaussian prim '{gaussian_prim_path}'.")

    def value(name: str) -> np.ndarray:
        attr = prim.GetAttribute(name)
        if not attr.IsValid() or not attr.HasValue():
            raise ValueError(f"Gaussian prim '{prim.GetPath()}' is missing '{name}'.")
        return np.asarray(attr.Get(), dtype=np.float32)

    xyz = value("positions")
    scales = value("scales").copy()
    opacity = np.clip(value("opacities").reshape(-1), 1.0e-6, 1.0 - 1.0e-6)
    orientations = prim.GetAttribute("orientations").Get()
    rotations = np.asarray([[q.GetReal(), *q.GetImaginary()] for q in orientations], dtype=np.float32)
    sh = value("radiance:sphericalHarmonicsCoefficients").reshape(len(xyz), -1, 3)
    if xyz.shape != scales.shape or len(rotations) != len(xyz) or sh.shape[1] < 1:
        raise ValueError("Gaussian attributes have incompatible lengths.")

    if target_max_extent is not None:
        extent = float(np.ptp(xyz, axis=0).max())
        if extent <= 0.0:
            raise ValueError("Gaussian field has zero extent.")
        scale = target_max_extent / extent
        xyz = (xyz - xyz.mean(axis=0, keepdims=True)) * scale
        scales *= scale
    if source_y_up:
        xyz = np.stack((xyz[:, 0], -xyz[:, 2], xyz[:, 1]), axis=1)
        scales = scales[:, (0, 2, 1)]
        # Left-multiply by the +90 degree X-axis rotation in wxyz convention.
        w, x, y, z = rotations.T
        root_half = np.float32(2.0**-0.5)
        rotations = np.stack(
            (root_half * (w - x), root_half * (w + x), root_half * (y - z), root_half * (y + z)), axis=1
        )
    rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(len(xyz), -1)
    fields = [(name, "<f4") for name in ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2")]
    fields += [(f"f_rest_{i}", "<f4") for i in range(rest.shape[1])]
    fields += [("opacity", "<f4"), ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4")]
    fields += [(f"rot_{i}", "<f4") for i in range(4)]
    vertices = np.empty(len(xyz), dtype=np.dtype(fields))
    for index, axis in enumerate("xyz"):
        vertices[axis] = xyz[:, index]
        vertices[f"f_dc_{index}"] = sh[:, 0, index]
        vertices[f"scale_{index}"] = np.log(np.maximum(scales[:, index], 1.0e-12))
    for index in range(rest.shape[1]):
        vertices[f"f_rest_{index}"] = rest[:, index]
    vertices["nx"] = vertices["ny"] = vertices["nz"] = 0.0
    vertices["opacity"] = np.log(opacity / (1.0 - opacity))
    for index in range(4):
        vertices[f"rot_{index}"] = rotations[:, index]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    properties = "\n".join(f"property float {name}" for name in vertices.dtype.names)
    with output_path.open("wb") as file:
        file.write(
            f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n{properties}\nend_header\n".encode()
        )
        vertices.tofile(file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="+", help="Gaussian USD assets.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported PLY files.")
    parser.add_argument("--gaussian-prim-path", default=None, help="Optional Gaussian prim path shared by all inputs.")
    parser.add_argument(
        "--target-max-extent", type=float, default=0.20, help="Normalize the largest extent to this size [m]."
    )
    parser.add_argument("--source-z-up", action="store_true", help="Do not apply the task's Y-up to Z-up conversion.")
    args = parser.parse_args()
    for asset in args.assets:
        output = Path(args.output_dir) / f"{Path(asset).stem}.ply"
        export_gaussian_usd_to_ply(
            asset,
            output,
            args.gaussian_prim_path,
            target_max_extent=args.target_max_extent,
            source_y_up=not args.source_z_up,
        )
        print(f"exported {output}")


if __name__ == "__main__":
    main()
