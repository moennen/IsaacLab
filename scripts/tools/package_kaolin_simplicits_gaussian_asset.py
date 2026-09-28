# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bake Gaussian-twin USD packages into Kaolin RKPM Simplicits assets.

The output preserves the original ``ParticleField3DGaussianSplat`` attributes
and adds Kaolin's ``SkinnedPhysicsPoints`` data on that field.  It can then be
used by the Gaussian-twin task's ``--simplicits-simulation`` profile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pxr import Gf, Sdf, Usd, UsdGeom


def _kaolin_api():
    try:
        from kaolin.io import usd as kaolin_usd
        from kaolin.physics.simplicits import PhysicsPoints, SimplicitsObject
    except ImportError as error:
        raise ImportError(
            "Kaolin is required for RKPM baking. Install the local checkout with "
            "`uv pip install --python .venv/bin/python -e ../kaolin --no-deps`."
        ) from error
    return kaolin_usd, PhysicsPoints, SimplicitsObject


def _find_gaussian(stage: Usd.Stage):
    gaussian = next((prim for prim in stage.Traverse() if prim.GetTypeName() == "ParticleField3DGaussianSplat"), None)
    if gaussian is None:
        raise ValueError(f"'{stage.GetRootLayer().identifier}' has no ParticleField3DGaussianSplat prim.")
    return gaussian


def _sample_indices(count: int, limit: int, seed: int, device: str) -> torch.Tensor:
    if limit >= count:
        return torch.arange(count, device=device, dtype=torch.long)
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.choice(count, size=limit, replace=False), device=device, dtype=torch.long)


def _copy_gaussian(stage: Usd.Stage, source, positions: np.ndarray, max_gaussians: int | None):
    root = stage.DefinePrim("/KaolinSimplicitsGaussianAsset", "Xform")
    stage.SetDefaultPrim(root)
    stage.DefinePrim("/KaolinSimplicitsGaussianAsset/visual", "Scope")
    destination = stage.DefinePrim("/KaolinSimplicitsGaussianAsset/visual/gaussians_0", "ParticleField3DGaussianSplat")
    source_count = len(source.GetAttribute("positions").Get())
    for source_attr in source.GetAttributes():
        name = source_attr.GetName()
        if name == "positions":
            destination.CreateAttribute(name, Sdf.ValueTypeNames.Point3fArray).Set(
                [Gf.Vec3f(*map(float, point)) for point in positions]
            )
            continue
        value = source_attr.Get()
        if value is None:
            continue
        if max_gaussians is not None and hasattr(value, "__len__"):
            if len(value) == source_count:
                value = value[:max_gaussians]
            elif source_count and len(value) % source_count == 0:
                value = value[: max_gaussians * (len(value) // source_count)]
        destination.CreateAttribute(name, source_attr.GetTypeName(), custom=source_attr.IsCustom()).Set(value)
    return destination


def package_asset(
    input_path: Path,
    output_path: Path,
    *,
    physics_points: int,
    quadrature_points: int,
    handles: int,
    rkpm_nodes: int,
    rkpm_points: int | None,
    youngs_modulus: float,
    poisson_ratio: float,
    density: float,
    max_gaussians: int | None,
    seed: int,
    device: str,
) -> None:
    """Bake one source package using Kaolin's documented RKPM constructor."""
    kaolin_usd, PhysicsPoints, SimplicitsObject = _kaolin_api()
    source_stage = Usd.Stage.Open(str(input_path))
    if source_stage is None:
        raise FileNotFoundError(f"Cannot open source asset '{input_path}'.")
    source = _find_gaussian(source_stage)
    positions = np.asarray(source.GetAttribute("positions").Get(), dtype=np.float32).reshape(-1, 3)
    if max_gaussians is not None:
        if max_gaussians <= 0:
            raise ValueError("--max-gaussians must be positive.")
        positions = positions[:max_gaussians]
    if len(positions) < 4:
        raise ValueError(f"'{input_path}' has too few Gaussian positions for RKPM ({len(positions)}).")
    indices = _sample_indices(len(positions), min(physics_points, len(positions)), seed, device)
    points = torch.as_tensor(np.ascontiguousarray(positions.copy()), device=device, dtype=torch.float32)
    physics = points.index_select(0, indices).contiguous()
    extent = torch.clamp(points.max(dim=0).values - points.min(dim=0).values, min=1.0e-6)
    material = PhysicsPoints(
        pts=physics,
        yms=float(youngs_modulus),
        prs=float(poisson_ratio),
        rhos=float(density),
        appx_vol=float(torch.prod(extent).item()),
    )
    simplicits = SimplicitsObject.create_with_rkpm(
        physics_points=material,
        num_handles=int(handles),
        num_nodes=int(rkpm_nodes),
        num_points=rkpm_points,
    )
    baked = simplicits.bake(num_qps=min(int(quadrature_points), len(physics)), renderable_pts=points)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Re-running a long batch after changing an RKPM preset should be safe:
    # replace only this generated layer rather than failing at CreateNew.
    stage = Usd.Stage.Open(str(output_path)) if output_path.exists() else Usd.Stage.CreateNew(str(output_path))
    if stage is None:
        raise RuntimeError(f"Cannot create or reopen output asset '{output_path}'.")
    if output_path.exists():
        stage.GetRootLayer().Clear()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    gaussian = _copy_gaussian(stage, source, positions, max_gaussians)
    gaussian.CreateAttribute("kaolin:simplicits:sourceGaussianUsd", Sdf.ValueTypeNames.String).Set(str(input_path))
    gaussian.CreateAttribute("kaolin:simplicits:method", Sdf.ValueTypeNames.Token).Set("rkpm")
    gaussian.CreateAttribute("kaolin:simplicits:numHandles", Sdf.ValueTypeNames.Int).Set(int(handles))
    gaussian.CreateAttribute("kaolin:simplicits:numPhysicsPoints", Sdf.ValueTypeNames.Int).Set(int(physics.shape[0]))
    gaussian.CreateAttribute("kaolin:simplicits:numQuadraturePoints", Sdf.ValueTypeNames.Int).Set(
        min(int(quadrature_points), len(physics))
    )
    kaolin_usd.add_skinned_physics(stage, gaussian, baked, instance_name="default", overwrite=True)
    stage.Save()


def _output_name(source: Path) -> str:
    name = source.name
    for suffix in ("_skinned_vbd_tet.usda", "_package.usda"):
        if name.endswith(suffix):
            return f"{name[: -len(suffix)]}_simplicits_rkpm.usda"
    return f"{source.stem}_simplicits_rkpm.usda"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="One Gaussian-twin package to convert.")
    source.add_argument("--input-dir", type=Path, help="Convert every baked.*_package.usda in this directory.")
    parser.add_argument("--output", type=Path, help="Output path for --input.")
    parser.add_argument("--output-dir", type=Path, help="Output directory (defaults to --input-dir).")
    parser.add_argument("--physics-points", type=int, default=4096)
    parser.add_argument("--quadrature-points", type=int, default=2048)
    parser.add_argument("--handles", type=int, default=16)
    parser.add_argument("--rkpm-nodes", type=int, default=128)
    parser.add_argument("--rkpm-points", type=int, default=4096)
    parser.add_argument("--youngs-modulus", type=float, default=1.0e5)
    parser.add_argument("--poisson-ratio", type=float, default=0.35)
    parser.add_argument("--density", type=float, default=300.0)
    parser.add_argument("--max-gaussians", type=int, default=None, help="Debug: retain only the first N splats.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.input is not None and args.output is None:
        parser.error("--input requires --output.")
    if args.input_dir is not None and args.output is not None:
        parser.error("--output is only valid with --input; use --output-dir with --input-dir.")
    return args


def main() -> None:
    args = _parse_args()
    if args.input is not None:
        work = [(args.input, args.output)]
    else:
        inputs = sorted(args.input_dir.glob("baked.*_package.usda"))
        if not inputs:
            raise ValueError(f"No baked.*_package.usda files found in '{args.input_dir}'.")
        output_dir = args.output_dir or args.input_dir
        work = [(source, output_dir / _output_name(source)) for source in inputs]
    for source, output in work:
        print(f"RKPM baking {source.name} -> {output}")
        package_asset(
            source,
            output,
            physics_points=args.physics_points,
            quadrature_points=args.quadrature_points,
            handles=args.handles,
            rkpm_nodes=args.rkpm_nodes,
            rkpm_points=args.rkpm_points,
            youngs_modulus=args.youngs_modulus,
            poisson_ratio=args.poisson_ratio,
            density=args.density,
            max_gaussians=args.max_gaussians,
            seed=args.seed,
            device=args.device,
        )
    print(f"Packaged {len(work)} Kaolin RKPM Simplicits asset(s).")


if __name__ == "__main__":
    main()
