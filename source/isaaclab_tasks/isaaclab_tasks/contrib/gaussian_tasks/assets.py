# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download and verify Gaussian-task bundles, or build a staged-upload manifest.

Nucleus is read-only here. Existing matching files are reused; mismatches require
--replace. Local sources support testing the exact upload bundle before publishing.
"""

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from pxr import Sdf, Usd, Vt

DEFAULT_SOURCE = "omniverse://content.ov.nvidia.com/Users/nicolasm@nvidia.com/isaaclab_assets/tasks"
MANIFEST = "task_assets_manifest.usda"
TASKS = ("berry", "tomato", "gaussian_twin")
USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}


def digest(path: Path) -> str:
    """Compute a file's SHA256 without loading large USDs into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(name: str) -> PurePosixPath:
    """Reject unsafe or unrecognized bundle paths."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts or path.parts[0] not in TASKS:
        raise ValueError(f"Unsafe task asset path: {name!r}")
    return path


def copy_source(source: str, relative: str, target: Path):
    """Copy one local, HTTP(S), or Nucleus file to a temporary local destination."""
    url = source.rstrip("/") + "/" + relative
    if source.startswith("omniverse://"):
        from isaaclab.utils.assets import _get_omni_client

        client = _get_omni_client()
        result = client.copy(url, str(target), client.CopyBehavior.OVERWRITE)
        if result != client.Result.OK:
            raise RuntimeError(f"Nucleus copy failed ({result}): {url}. Check upload/access permissions.")
    elif source.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as out:
            shutil.copyfileobj(response, out)
    elif "://" in source:
        raise ValueError(f"Unsupported source: {source}")
    else:
        shutil.copyfile(Path(source).expanduser() / relative, target)


def write_manifest(path: Path, entries: list[dict]):
    """Author the download inventory as typed USD arrays."""
    stage = Usd.Stage.CreateNew(str(path))
    root = stage.DefinePrim("/TaskAssets", "Scope")
    stage.SetDefaultPrim(root)
    root.CreateAttribute("bundle:schemaVersion", Sdf.ValueTypeNames.Int).Set(2)
    root.CreateAttribute("bundle:paths", Sdf.ValueTypeNames.StringArray).Set([e["path"] for e in entries])
    root.CreateAttribute("bundle:bytes", Sdf.ValueTypeNames.UInt64Array).Set(
        Vt.UInt64Array([e["bytes"] for e in entries])
    )
    root.CreateAttribute("bundle:sha256", Sdf.ValueTypeNames.StringArray).Set([e["sha256"] for e in entries])
    stage.GetRootLayer().Save()


def read_manifest(path: Path) -> list[dict]:
    """Read and validate a schema-v2 USD download inventory."""
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    if root.GetAttribute("bundle:schemaVersion").Get() != 2:
        raise ValueError("Unsupported asset manifest schema; publish the USD-only v2 bundle")
    return [
        dict(path=p, bytes=b, sha256=s)
        for p, b, s in zip(
            root.GetAttribute("bundle:paths").Get(),
            root.GetAttribute("bundle:bytes").Get(),
            root.GetAttribute("bundle:sha256").Get(),
            strict=True,
        )
    ]


def build_manifest(root: Path):
    """Inventory staged assets. Refuse symlinks and existing manifests."""
    output = root / MANIFEST
    if output.exists():
        raise FileExistsError(f"Manifest already exists: {output}")
    entries = []
    for task in TASKS:
        if not (root / task).is_dir():
            raise FileNotFoundError(root / task)
        for path in sorted((root / task).rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Asset bundle must not contain symlinks: {path}")
            if path.is_file():
                if path.suffix not in USD_SUFFIXES:
                    raise ValueError(f"Only USD/USDZ files may be published: {path}")
                name = path.relative_to(root).as_posix()
                safe_path(name)
                entries.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path)})
    write_manifest(output, entries)
    print(f"Manifest: {len(entries)} files, {sum(e['bytes'] for e in entries):,} bytes")


def download(source: str, destination: Path, task: str, replace: bool = False):
    """Verify and populate a local task cache from a published manifest."""
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task-download-", dir=destination) as temporary:
        temporary = Path(temporary)
        manifest = temporary / MANIFEST
        copy_source(source, MANIFEST, manifest)
        entries = read_manifest(manifest)
        names = [str(safe_path(e["path"])) for e in entries]
        if any(PurePosixPath(name).suffix not in USD_SUFFIXES for name in names):
            raise ValueError("Manifest contains a non-USD asset")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate asset paths in manifest")
        selected = [e for e in entries if task == "all" or PurePosixPath(e["path"]).parts[0] == task]
        if not selected:
            raise ValueError(f"No assets found for {task}")
        for index, entry in enumerate(selected, 1):
            target = destination / entry["path"]
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Destination escapes cache through a symlink: {target}")
            if target.exists():
                if target.stat().st_size == entry["bytes"] and digest(target) == entry["sha256"]:
                    continue
                if not replace:
                    raise FileExistsError(f"Mismatched local file: {target}; use --replace to replace it.")
            partial = temporary / "download"
            print(f"[{index}/{len(selected)}] {entry['path']}", flush=True)
            copy_source(source, entry["path"], partial)
            if partial.stat().st_size != entry["bytes"] or digest(partial) != entry["sha256"]:
                raise ValueError(f"Asset checksum mismatch: {entry['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial.replace(target)
    print(f"Verified {len(selected)} files under {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(os.environ.get("ISAACLAB_TASK_ASSET_ROOT", str(Path.home() / ".cache/isaaclab/tasks"))),
    )
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--replace", action="store_true", help="Replace mismatched cached files after verification")
    parser.add_argument(
        "--manifest", type=Path, help="Build the upload manifest in this already-staged tasks directory"
    )
    args = parser.parse_args()
    if args.manifest:
        build_manifest(args.manifest)
    else:
        download(args.source, args.destination.expanduser(), args.task, args.replace)


if __name__ == "__main__":
    main()
