# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that a Gaussian-task USD-only upload and its dependencies are self-contained."""

import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

from pxr import UsdUtils

import isaaclab  # noqa: F401  # Register the usd-exchange plugin search path before pxr.

BUILTIN_MDL = {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl", "ParticleFieldEmissive.mdl"}


def validate(root: Path) -> dict:
    """Validate external file references without importing task code or simulating."""
    root = root.resolve()
    errors = []
    checked = 0

    def reference(owner, value):
        nonlocal checked
        if not value or value in BUILTIN_MDL:
            return
        checked += 1
        # USD package-relative syntax: check the outer archive here.
        value = value.split("[", 1)[0]
        target = (owner.parent / value).resolve()
        if "://" in value or Path(value).is_absolute() or not target.is_relative_to(root) or not target.is_file():
            errors.append(f"{owner.relative_to(root)} -> {value}")

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix not in {".usd", ".usda", ".usdc", ".usdz"}:
            errors.append(f"Non-USD published file: {path.relative_to(root)}")
        if path.suffix in {".usd", ".usda", ".usdc"}:
            for dependencies in UsdUtils.ExtractExternalReferences(str(path)):
                for value in dependencies:
                    reference(path, value)
        elif path.suffix == ".xml":
            tree = ET.parse(path)
            compiler = tree.find("compiler")
            for element in tree.iter():
                value = element.get("file")
                if value:
                    directory = ""
                    if compiler is not None and element.tag in {"mesh", "texture"}:
                        directory = compiler.get(f"{element.tag}dir", compiler.get("assetdir", ""))
                    reference(path, str(Path(directory) / value))
        elif path.suffix == ".mdl":
            for value in re.findall(r'texture_2d\("([^\"]+)"', path.read_text()):
                reference(path, value)
        elif path.suffix == ".usdz":
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    errors.append(f"Corrupt USDZ: {path}")
                members = set(archive.namelist())
                for member in members:
                    if PurePosixPath(member).suffix in {".npz", ".json", ".xml", ".py", ".glb"}:
                        errors.append(f"Legacy sidecar inside USDZ: {path.name}[{member}]")
                for member in members:
                    if PurePosixPath(member).suffix not in {".usd", ".usda", ".usdc"}:
                        continue
                    for dependencies in UsdUtils.ExtractExternalReferences(f"{path}[{member}]"):
                        for value in dependencies:
                            if value in BUILTIN_MDL:
                                continue
                            checked += 1
                            name = str(PurePosixPath(member).parent / value)
                            if name not in members:
                                errors.append(f"{path.name}[{member}] -> {value}")
    if errors:
        raise ValueError("Nonportable/missing asset references:\n" + "\n".join(errors))
    return {"root": str(root), "references_checked": checked, "passed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(validate(parser.parse_args().directory))
