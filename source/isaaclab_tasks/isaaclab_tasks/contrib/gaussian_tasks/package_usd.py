# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Migrate an original Gaussian-task bundle into self-contained USD/USDZ assets.

Authoring only. Requires the original local bundle and POC for the raspberry
repair; end users need only the generated USD files and this IsaacLab branch.
Never overwrites an existing destination or edits source assets.
"""

import argparse
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from pxr import Sdf, Usd, UsdUtils, Vt

from isaaclab_tasks.contrib.berry_pick.rebuild_appearance import rebuild_raspberry
from isaaclab_tasks.contrib.gaussian_tasks.usd_task_data import write_array, write_data


def attribution(stage: Usd.Stage, source: Path):
    """Preserve bundled license/readme notices as USD string metadata."""
    notices = {
        p.relative_to(source).as_posix(): p.read_text()
        for p in sorted(source.rglob("*"))
        if p.suffix.lower() in {".md", ".txt"}
    }
    write_data(stage, str(stage.GetDefaultPrim().GetPath()) + "/TaskData/Notices", notices)


def package(stage: Usd.Stage, output: Path):
    """Package flattened USD and its referenced textures/MDL without sidecars."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="usd-authoring-") as directory:
        layer = Path(directory) / "asset.usdc"
        stage.GetRootLayer().Export(str(layer))
        if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(layer)), str(output)):
            raise RuntimeError(f"USDZ packaging failed: {output}")


def berries(source: Path, output: Path, poc: Path):
    repaired, report = rebuild_raspberry(poc)
    print("Raspberry SH repair:", report, flush=True)
    for berry in ("raspberry", "blackberry", "blueberry", "strawberry"):
        folder = source / "berry" / berry
        stage = Usd.Stage.Open(Usd.Stage.Open(str(folder / f"{berry}.usdc")).Flatten())
        root = stage.GetDefaultPrim()
        for name in ("berry:profile", "berry:proxy", "berry:splats"):
            root.RemoveProperty(name)
        root.CreateAttribute("berry:schemaVersion", Sdf.ValueTypeNames.Int).Set(2)
        gaussians = stage.GetPrimAtPath("/Berry/Gaussians")
        with np.load(folder / "splats.npz") as splats:
            keys = [k for k in splats.files if k not in {"xyz", "scales", "rotations", "alpha", "sh"}]
            gaussians.CreateAttribute("berry:arrayKeys", Sdf.ValueTypeNames.StringArray).Set(keys)
            for key in keys:
                write_array(gaussians, "berry:arrays:" + key, splats[key])
        if berry == "raspberry":
            gaussians.GetAttribute("radiance:sphericalHarmonicsCoefficients").Set(
                Vt.Vec3fArray.FromNumpy(repaired.reshape(-1, 3))
            )
            write_data(stage, "/Berry/TaskData/AppearanceRepair", report)
        tissue = stage.GetPrimAtPath("/Berry/Tissue")
        with np.load(folder / "proxy.npz") as proxy:
            keys = [k for k in proxy.files if k != "xyz"]
            tissue.CreateAttribute("berry:arrayKeys", Sdf.ValueTypeNames.StringArray).Set(keys)
            for key in keys:
                write_array(tissue, "berry:arrays:" + key, proxy[key])
        write_data(stage, "/Berry/TaskData/Profile", json.loads((folder / "profile.json").read_text()))
        attribution(stage, folder)
        package(stage, output / "berry" / berry / f"{berry}.usdz")


def tomatoes(source: Path, output: Path):
    folder = source / "tomato/plant"
    stage = Usd.Stage.Open(Usd.Stage.Open(str(folder / "plant_visual.usdc")).Flatten())
    root = stage.GetDefaultPrim()
    path = str(root.GetPath())
    # Replace asset-path metadata referring to former JSON sidecars.
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath) and Path(value.path).suffix in {".json", ".npz", ".xml"}:
                prim.RemoveProperty(attr.GetName())
    manifest = json.loads((folder / "plant_manifest.json").read_text())
    for filename, name in (
        ("plant_manifest", "Manifest"),
        ("mapping_names", "Bindings"),
        ("mechanical_calibration", "Calibration"),
    ):
        write_data(stage, f"{path}/TaskData/{name}", json.loads((folder / f"{filename}.json").read_text()))
    areas = []
    objects = {obj["id"]: obj for obj in manifest["objects"]}
    for leaf in manifest["leaves"]:
        with np.load(folder / objects[leaf["id"]]["mesh_npz"]) as mesh:
            triangles = mesh["vertices"][mesh["faces"]]
            areas.append(
                float(
                    np.linalg.norm(
                        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
                    ).sum()
                    / 4
                )
            )
    write_data(stage, path + "/TaskData/LeafAreas", areas)
    xml = (folder / "plant_physics.xml").read_text()
    if any(e.get("file") for e in ET.fromstring(xml).iter()):
        raise ValueError("Plant MJCF must contain its geometry inline")
    root.CreateAttribute("tomato:mjcf", Sdf.ValueTypeNames.String).Set(xml)
    attribution(stage, folder)
    package(stage, output / "tomato/plant/plant_visual.usdz")
    for background in ("greenhouse", "greenhouse_modeled"):
        folder = source / "tomato" / background
        stage = Usd.Stage.Open(Usd.Stage.Open(str(folder / "greenhouse.usda")).Flatten())
        attribution(stage, folder)
        package(stage, output / "tomato" / background / "greenhouse.usdz")
    shutil.copy2(source / "tomato/greenhouse_modeled/lighting.usda", output / "tomato/greenhouse_modeled/lighting.usda")


def twins(source: Path, output: Path):
    # Existing twin assets already store all physics/appearance data in USD.
    # Keep their public names (including the paired optional Simplicits assets).
    for path in sorted((source / "gaussian_twin").rglob("*")):
        if path.suffix in {".usd", ".usda", ".usdc", ".usdz"}:
            target = output / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    notices = Usd.Stage.CreateNew(str(output / "gaussian_twin/attribution.usdc"))
    notices.SetDefaultPrim(notices.DefinePrim("/Attribution", "Scope"))
    attribution(notices, source / "gaussian_twin")
    notices.GetRootLayer().Save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poc", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    args.output.mkdir(parents=True)
    berries(args.source, args.output, args.poc)
    tomatoes(args.source, args.output)
    twins(args.source, args.output)
    print(f"Packaged USD-only bundle: {args.output}", flush=True)


if __name__ == "__main__":
    main()
