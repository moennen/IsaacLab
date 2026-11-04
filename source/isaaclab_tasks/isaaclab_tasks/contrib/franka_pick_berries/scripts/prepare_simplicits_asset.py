# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bake Kaolin Simplicits and MLS contact data into a self-contained raspberry USDZ."""

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def _surface(vertices, tets):
    """Extract outward boundary faces independently of the input tetrahedron winding."""
    tets = np.asarray(tets).copy()
    corners = np.asarray(vertices, dtype=np.float64)[tets]
    edges = corners[:, 1:] - corners[:, :1]
    volume6 = np.einsum("ij,ij->i", np.cross(edges[:, 0], edges[:, 1]), edges[:, 2])
    tolerance = 10 * np.finfo(np.float64).eps * np.linalg.norm(edges, axis=2).max(axis=1) ** 3
    if np.any(~np.isfinite(volume6)) or np.any(np.abs(volume6) <= tolerance):
        raise ValueError("Cannot bake contact normals from degenerate tetrahedra")
    negative = volume6 < 0
    tets[negative] = tets[negative][:, [1, 0, 2, 3]]
    faces = tets[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Cannot bake contact normals from a nonmanifold tetrahedral mesh")
    return faces[first[counts == 1]]


def prepare(args):
    """Train once and retain the original package resources while adding baked USD APIs."""
    import torch
    from kaolin.io.usd import add_skinned_physics
    from kaolin.physics.simplicits import PhysicsPoints, SimplicitsObject, SkinnedPhysicsPoints, sample_mesh_surface

    from pxr import Sdf, Usd, UsdGeom, Vt

    from isaaclab_tasks.contrib.franka_pick_berries.assets.asset_paths import raspberry_asset_path
    from isaaclab_tasks.contrib.franka_pick_berries.assets.berry_asset import load_berry_asset, load_berry_tetmesh
    from isaaclab_tasks.contrib.franka_pick_berries.physics.materials import POISSON_RATIO, YOUNG_MODULUS

    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(args.seed)
    stage, gaussians, particles, profile = load_berry_asset(args.input or raspberry_asset_path())
    package = stage.GetRootLayer().realPath.split("[")[0]
    if not zipfile.is_zipfile(package):
        raise ValueError("Simplicits preparation expects a USDZ source to preserve its packaged shader resources")
    if args.quadrature_points > len(particles["xyz"]):
        raise ValueError("Quadrature count cannot exceed the source tissue sample count")
    device = args.device
    pts = torch.as_tensor(particles["xyz"], dtype=torch.float32, device=device)
    volume = float(particles["particle_volume"]) * len(pts)
    material = PhysicsPoints(pts, YOUNG_MODULUS, POISSON_RATIO, float(profile["simulation"]["density"]), volume)
    if args.basis == "rkpm":
        model = SimplicitsObject.create_with_rkpm(
            material,
            num_handles=args.handles,
            num_nodes=args.rkpm_nodes,
            num_points=min(1000, len(pts)),
        )
    else:
        model = SimplicitsObject.create_with_mlp(
            material,
            num_handles=args.handles,
            num_samples=min(1000, len(pts)),
            model_layers=2,
            training_num_steps=args.training_steps,
            training_log_every=500,
            display_progress=True,
        )
    baked = model.bake(
        num_qps=args.quadrature_points,
        renderable_pts=torch.as_tensor(gaussians["xyz"], dtype=torch.float32, device=device),
        seed=args.seed,
    )
    add_skinned_physics(stage, "/Berry/Gaussians", baked, overwrite=True)
    mesh = load_berry_tetmesh(stage)
    vertices = torch.as_tensor(mesh["xyz"], device=device)
    faces = torch.as_tensor(_surface(mesh["xyz"], mesh["tets"]), dtype=torch.long, device=device)
    samples, normals = sample_mesh_surface(vertices, faces, args.contact_points, seed=args.seed, device=device)
    fv = vertices[faces]
    area = torch.linalg.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0]).norm(dim=1).sum() / 2
    contact = SkinnedPhysicsPoints.from_skinning_mod(
        samples,
        torch.full_like(samples[:, 0], YOUNG_MODULUS),
        torch.full_like(samples[:, 0], POISSON_RATIO),
        torch.full_like(samples[:, 0], float(profile["simulation"]["density"])),
        torch.tensor(volume, device=device),
        model.skinning_mod,
    )
    from isaaclab_tasks.contrib.franka_pick_berries.assets.simplicits import COLLISION_PATH

    prim = UsdGeom.Points.Define(stage, COLLISION_PATH)
    prim.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(samples.cpu().numpy()))
    prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.cpu().numpy()))
    prim.GetPrim().CreateAttribute("berry:restAreas", Sdf.ValueTypeNames.FloatArray).Set(
        Vt.FloatArray.FromNumpy(np.full(len(samples), float(area) / len(samples), np.float32))
    )
    add_skinned_physics(stage, prim.GetPrim(), contact, overwrite=True)
    stage.GetDefaultPrim().CreateAttribute("berry:simplicitsVersion", Sdf.ValueTypeNames.Int).Set(1)
    stage.GetDefaultPrim().CreateAttribute("berry:simplicitsBasis", Sdf.ValueTypeNames.Token).Set(args.basis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep relative MDL and texture references intact and use USD's aligned ZIP writer.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory, zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        archive.extractall(directory)
        root = Path(directory) / names[0]
        stage.GetRootLayer().Export(str(root))
        prepared = Path(directory) / "simplicits-package.usdz"
        with Sdf.ZipFileWriter.CreateNew(str(prepared)) as writer:
            for name in names:
                writer.AddFile(str(Path(directory) / name), name)
        from isaaclab_tasks.contrib.franka_pick_berries.assets.simplicits import load_simplicits

        data = load_simplicits(Usd.Stage.Open(str(prepared)))
        if len(data["pts"]) != args.quadrature_points:
            raise ValueError("The prepared package did not retain the requested quadrature count")
        if args.output.exists():
            raise FileExistsError(args.output)
        prepared.replace(args.output)
    print(f"Wrote {args.output}: {len(data['pts'])} quadrature points, {data['skinning_weights'].shape[1]} handles")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Schema-v2 raspberry USDZ source; defaults to the Isaac Lab asset root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--basis", choices=("rkpm", "mlp"), default="rkpm")
    parser.add_argument("--rkpm_nodes", type=int, default=32)
    parser.add_argument("--handles", type=int, default=8)
    parser.add_argument("--quadrature_points", type=int, default=512)
    parser.add_argument("--contact_points", type=int, default=256)
    parser.add_argument("--training_steps", type=int, default=2000, help="Training iterations for --basis mlp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.handles, args.quadrature_points, args.contact_points) <= 0:
        parser.error("Training and sampling counts must be positive")
    if args.basis == "mlp" and args.training_steps <= 0:
        parser.error("MLP training steps must be positive")
    if args.basis == "rkpm" and args.handles > 1 and args.rkpm_nodes < 3 * (args.handles + 1):
        parser.error("RKPM requires at least 3 * (handles + 1) nodes")
    prepare(args)


if __name__ == "__main__":
    main()
