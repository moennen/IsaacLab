# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local transport and relocated installer tests for the Gaussian-task tooling."""

import os
import subprocess
from pathlib import Path

import pytest

from isaaclab_tasks.contrib.gaussian_tasks import assets
from isaaclab_tasks.contrib.gaussian_tasks.add_ebc_background import add_ebc_background


def test_setup_resolves_repository_from_another_working_directory(tmp_path):
    # Stop at the first dependency command: exercise path resolution without installing anything.
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$UV_PROJECT_ENVIRONMENT"\nexit 73\n')
    uv.chmod(0o755)
    setup = Path(assets.__file__).with_name("setup.sh")
    result = subprocess.run(
        ["bash", str(setup), "--renderer-internal"],
        cwd=tmp_path,
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=10,
    )
    root = Path(__file__).resolve().parents[3]
    assert result.returncode == 73, result.stderr
    assert result.stdout.splitlines() == [str(root), str(root / ".venv-tasks")]


def bundle(tmp_path):
    source = tmp_path / "source"
    for task in assets.TASKS:
        (source / task).mkdir(parents=True)
        (source / task / "sample.usda").write_text("#usda 1.0\n")
    assets.build_manifest(source)
    return source


def test_add_ebc_preserves_scan_and_existing_inventory(tmp_path):
    from pxr import Usd, UsdUtils

    target = bundle(tmp_path)
    original = (target / assets.MANIFEST).read_bytes()
    source = tmp_path / "EBC_Newton_Demo"
    source.mkdir()
    scan = source / "point_cloud.usd"
    stage = Usd.Stage.CreateNew(str(scan))
    stage.SetDefaultPrim(stage.DefinePrim("/GaussianSplats", "Xform"))
    stage.GetRootLayer().Save()
    aligned = Usd.Stage.CreateNew(str(source / "point_cloud_aligned_new.usda"))
    aligned.DefinePrim("/World", "Xform")
    aligned.DefinePrim("/World/GaussianBackground", "Xform").GetReferences().AddReference(
        "point_cloud.usd", "/GaussianSplats"
    )
    aligned.GetRootLayer().Save()
    output = add_ebc_background(source, target)
    assert output.with_name("point_cloud.usd").read_bytes() == scan.read_bytes()
    assert Usd.Stage.Open(str(output)).GetDefaultPrim().GetName() == "World"
    assert UsdUtils.ExtractExternalReferences(str(output)) == ([], ["point_cloud.usd"], [])
    assert (tmp_path / "source-manifest-before-ebc.usda").read_bytes() == original
    entries = assets.read_manifest(target / assets.MANIFEST)
    assert len(entries) == len(assets.TASKS) + 2
    for entry in entries:
        assert assets.digest(target / entry["path"]) == entry["sha256"]
    with pytest.raises(FileExistsError):
        add_ebc_background(source, target)


def test_download_is_selective_verified_and_idempotent(tmp_path):
    source = bundle(tmp_path)
    target = tmp_path / "cache"
    assets.download(str(source), target, "berry")
    assert (target / "berry/sample.usda").read_bytes() == b"#usda 1.0\n"
    assert not (target / "tomato").exists()
    assets.download(str(source), target, "berry")
    (target / "berry/sample.usda").write_bytes(b"changed")
    with pytest.raises(FileExistsError):
        assets.download(str(source), target, "berry")
    assets.download(str(source), target, "berry", replace=True)
    assert (target / "berry/sample.usda").read_bytes() == b"#usda 1.0\n"


def test_corrupt_source_does_not_replace_local_file(tmp_path):
    source = bundle(tmp_path)
    target = tmp_path / "cache"
    assets.download(str(source), target, "berry")
    (target / "berry/sample.usda").write_bytes(b"local")
    (source / "berry/sample.usda").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        assets.download(str(source), target, "berry", replace=True)
    assert (target / "berry/sample.usda").read_bytes() == b"local"


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "berry/../../escape", "berry\\escape"])
def test_rejects_unsafe_manifest_paths(tmp_path, name):
    source = bundle(tmp_path)
    manifest = source / assets.MANIFEST
    entries = assets.read_manifest(manifest)
    entries[0]["path"] = name
    assets.write_manifest(manifest, entries)
    with pytest.raises(ValueError, match="Unsafe"):
        assets.download(str(source), tmp_path / "cache", "all")


def test_rejects_destination_symlink_escape(tmp_path):
    source = bundle(tmp_path)
    target = tmp_path / "cache"
    outside = tmp_path / "outside"
    target.mkdir()
    outside.mkdir()
    (target / "berry").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        assets.download(str(source), target, "berry")
    assert not list(outside.iterdir())


def test_rejects_non_usd_upload_sidecars(tmp_path):
    for task in assets.TASKS:
        (tmp_path / task).mkdir()
    (tmp_path / "berry/profile.json").write_text("{}")
    with pytest.raises(ValueError, match="Only USD"):
        assets.build_manifest(tmp_path)
