# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactively align a referenced USD asset against an IsaacLab task scene.

The alignment output is a small USDA composition layer. It references the source asset and
authors only the wrapper transform, leaving the source file unchanged.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True, help="Registered IsaacLab task to load.")
parser.add_argument("--background-usd", type=Path, required=True, help="Source USD asset to reference.")
parser.add_argument(
    "--output-usd",
    type=Path,
    required=True,
    help="USDA alignment layer to write. A .usdz suffix is normalized to .usda.",
)
parser.add_argument(
    "--background-prim",
    type=str,
    default="/World/GaussianBackground",
    help="World-level prim path for the referenced asset.",
)
parser.add_argument("--num-envs", type=int, default=1, help="Number of task environments to display.")
parser.add_argument("--translation", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
parser.add_argument(
    "--rotation-degrees", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("ROLL", "PITCH", "YAW")
)
parser.add_argument("--scale", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("X", "Y", "Z"))
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def _make_transform_ops(prim):
    """Create the stable translate/orient/scale operation order on a wrapper prim."""
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    return (
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble),
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble),
        xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble),
    )


def _rotation_quaternion(angles_degrees):
    """Return an XYZ Euler rotation as a USD quaternion."""
    from pxr import Gf

    roll, pitch, yaw = angles_degrees
    rotation = (
        Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), roll)
        * Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), pitch)
        * Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw)
    )
    return rotation.GetQuat()


def _set_transform(ops, translation, rotation_degrees, scale):
    """Write the UI transform values to the live USD prim."""
    from pxr import Gf

    translate_op, orient_op, scale_op = ops
    translate_op.Set(Gf.Vec3d(*translation))
    orient_op.Set(_rotation_quaternion(rotation_degrees))
    scale_op.Set(Gf.Vec3d(*scale))


def _resolve_reference_roots(source_path: Path):
    """Return the opened source layer and prim paths to reference."""
    from pxr import Sdf, Usd

    source_path = source_path.resolve()
    source_layer = Sdf.Layer.FindOrOpen(str(source_path))
    if source_layer is None:
        raise ValueError(f"Unable to open USD asset: {source_path}")
    source_stage = Usd.Stage.Open(source_layer)
    default_prim = source_stage.GetDefaultPrim()
    root_paths = (
        [default_prim.GetPath()]
        if default_prim.IsValid()
        else [prim.GetPath() for prim in source_stage.GetPseudoRoot().GetChildren()]
    )
    if not root_paths:
        raise ValueError(f"USD asset has no root prims: {source_path}")
    return source_path, source_layer, root_paths


def _normalize_output_path(output_path: Path) -> Path:
    """Ensure the alignment layer uses a format supported by Sdf.Layer.CreateNew."""
    if output_path.suffix.lower() == ".usdz":
        return output_path.with_suffix(".usda")
    return output_path


def _add_asset_reference(prim, source_path: Path, root_paths) -> None:
    """Reference the source default prim, or all of its root prims if no default is authored."""
    source_path = source_path.resolve()
    for root_path in root_paths:
        prim.GetReferences().AddReference(str(source_path), root_path)


def _save_alignment(output_path: Path, source_path: Path, root_paths, prim_path: str, live_ops) -> None:
    """Save a minimal USDA reference wrapper containing only alignment data."""
    from pxr import Sdf, Usd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(output_path))
    stage = Usd.Stage.Open(layer)
    prim = stage.DefinePrim(prim_path, "Xform")
    source_asset = source_path.resolve()
    destination_dir = output_path.resolve().parent
    relative_source = Path(os.path.relpath(source_asset, destination_dir))
    for root_path in root_paths:
        prim.GetReferences().AddReference(relative_source.as_posix(), root_path)
    ops = _make_transform_ops(prim)
    # Copy the live USD op values rather than the numeric-panel state. This preserves edits made
    # with Kit's translate/rotate/scale gizmos as well as edits made through this tool's fields.
    for output_op, live_op in zip(ops, live_ops):
        output_op.Set(live_op.Get())
    stage.GetRootLayer().Save()


def _build_ui(prim, ops, source_path: Path, root_paths, output_path: Path):
    """Create the Kit alignment control window."""
    import omni.ui as ui

    def save(*_args):
        _save_alignment(output_path, source_path, root_paths, prim.GetPath().pathString, ops)
        status.text = f"Saved {output_path}"

    # ``omni.ui.Window`` is not itself a context manager in Isaac Sim 6; its frame is.
    window = ui.Window("Scene Alignment", width=420, height=260)
    import omni.usd

    omni.usd.get_context().get_selection().set_selected_prim_paths([prim.GetPath().pathString], False)
    with window.frame:
        with ui.VStack(spacing=8, height=0, padding=8):
            ui.Label("Referenced asset", tooltip=str(source_path))
            ui.Label(prim.GetPath().pathString)
            ui.Label("Use Kit's viewport transform gizmos, then save the alignment layer.")
            ui.Button("Save USDA", clicked_fn=save)
            status = ui.Label("Alignment layer not saved yet.")
    return window


def main() -> int:
    """Load the task and run the interactive alignment session."""
    import gymnasium as gym

    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.sim import get_current_stage

    import isaaclab_tasks  # noqa: F401

    if not args_cli.background_usd.is_file():
        parser.error(f"Background USD does not exist: {args_cli.background_usd}")
    if not args_cli.background_prim.startswith("/"):
        parser.error("--background-prim must be an absolute USD path.")
    if any(value <= 0.0 or not math.isfinite(value) for value in args_cli.scale):
        parser.error("--scale values must be finite and positive.")
    args_cli.output_usd = _normalize_output_path(args_cli.output_usd)

    app_launcher = AppLauncher(vars(args_cli))
    simulation_app = app_launcher.app

    env_cfg, _ = resolve_task_config(args_cli.task, None, overrides=hydra_args)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise TypeError(f"Task '{args_cli.task}' did not resolve to ManagerBasedRLEnvCfg.")
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.env_name = args_cli.task
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # The AppLauncher creates Kit before the task, but the USD stage is only available after
    # the environment has initialized its SimulationContext and scene.
    stage = get_current_stage()
    if stage is None:
        env.close()
        simulation_app.close()
        raise RuntimeError("Kit did not provide an active USD stage after environment initialization.")
    prim = stage.DefinePrim(args_cli.background_prim, "Xform")
    source_asset, _source_layer, root_paths = _resolve_reference_roots(args_cli.background_usd)
    _add_asset_reference(prim, source_asset, root_paths)
    ops = _make_transform_ops(prim)
    _set_transform(ops, args_cli.translation, args_cli.rotation_degrees, args_cli.scale)
    alignment_window = _build_ui(
        prim,
        ops,
        source_asset,
        root_paths,
        args_cli.output_usd,
    )

    try:
        # Pump Kit directly while the alignment window is open.  Calling ``env.sim.render()``
        # routes through the environment visualizer manager, which may remove the Kit visualizer
        # when the simulation is not playing and can make this render-only tool tear down early.
        while simulation_app.is_running() and not simulation_app.is_exiting():
            simulation_app.update()
    finally:
        alignment_window.visible = False
        env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
