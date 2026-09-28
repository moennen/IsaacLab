# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Add the correctly oriented EBC Gaussian room to an existing USD-only bundle."""

import argparse
import shutil
import tempfile
from pathlib import Path

from pxr import Sdf, UsdUtils

from isaaclab_tasks.contrib.gaussian_tasks.assets import MANIFEST, digest, read_manifest, write_manifest


def add_ebc_background(source: Path, bundle: Path) -> Path:
    """Copy the complete scan and alignment, preserving the previous inventory outside the bundle."""
    destination = bundle / "gaussian_twin/background/ebc"
    backup = bundle.parent / f"{bundle.name}-manifest-before-ebc.usda"
    if destination.exists() or backup.exists():
        raise FileExistsError("EBC destination or manifest backup already exists; refusing to overwrite it")
    scan = source / "point_cloud.usd"
    alignment = source / "point_cloud_aligned_new.usda"
    for path in (scan, alignment):
        if not path.is_file():
            raise FileNotFoundError(path)
    if any(UsdUtils.ExtractExternalReferences(str(scan))[i] for i in (0, 2)):
        raise ValueError("Expected a self-contained Gaussian scan without sublayers or payloads")
    references = UsdUtils.ExtractExternalReferences(str(scan))[1]
    if set(references) - {"ParticleFieldEmissive.mdl"}:
        raise ValueError(f"Unexpected scan dependencies: {references}")
    if UsdUtils.ExtractExternalReferences(str(alignment)) != ([], ["point_cloud.usd"], []):
        raise ValueError("Expected the alignment to reference only its neighboring point_cloud.usd")
    entries = read_manifest(bundle / MANIFEST)
    with tempfile.TemporaryDirectory(prefix="ebc-package-", dir=bundle.parent) as temporary:
        temporary = Path(temporary)
        staged = temporary / "ebc"
        staged.mkdir()
        shutil.copy2(scan, staged / scan.name)
        layer = Sdf.Layer.CreateNew(str(staged / "aligned.usda"))
        layer.TransferContent(Sdf.Layer.FindOrOpen(str(alignment)))
        # A layer defaultPrim is a root-prim name, not an absolute nested path.
        layer.defaultPrim = "World"
        layer.customLayerData = {"alignmentSource": alignment.name}
        layer.Save()
        for path in sorted(staged.iterdir()):
            entries.append(
                {
                    "path": (destination / path.name).relative_to(bundle).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
        write_manifest(temporary / MANIFEST, sorted(entries, key=lambda entry: entry["path"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / MANIFEST, backup)
        staged.rename(destination)
        (temporary / MANIFEST).replace(bundle / MANIFEST)
    return destination / "aligned.usda"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original EBC_Newton_Demo directory")
    parser.add_argument("--bundle", type=Path, required=True, help="Existing USD-only tasks directory")
    args = parser.parse_args()
    print(f"Added EBC background: {add_ebc_background(args.source.resolve(), args.bundle.resolve())}")


if __name__ == "__main__":
    main()
