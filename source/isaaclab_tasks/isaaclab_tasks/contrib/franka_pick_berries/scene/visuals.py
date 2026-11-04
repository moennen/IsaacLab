# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared USD visual scene for the berry workcell, independent of rendering backend."""

from pxr import Gf, Usd, UsdGeom, UsdShade

from ..gaussian_splats.render_delegate import author_berry_visual
from .room_scan import add_room_scan
from .tableware import add_tableware_visuals

VISUAL_ROOT_PATH = "/World/Visuals"


def author_scene_visuals(
    stage: Usd.Stage,
    tissues,
    builtin_material: bool = True,
    root_path: str = "/World",
) -> None:
    """Author berries, container, dishes and aligned room from the same assets and transforms."""
    tissues = list(tissues)
    for tissue in tissues:
        root = f"{root_path}/Berries/{tissue.name}"
        author_berry_visual(stage, tissue, root, builtin_material)
        field = UsdGeom.Xformable(stage.GetPrimAtPath(f"{root}/Gaussians/Field"))
        field.ClearXformOpOrder()
        field.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in tissue.offset)))
    add_tableware_visuals(stage, f"{root_path}/Tableware")
    shader = UsdShade.Shader.Get(stage, f"{root_path}/Berries/{tissues[0].name}/Materials/Radiance/Shader")
    add_room_scan(stage, shader, f"{root_path}/Room")


def visual_scene_cfg(specs):
    """Declare the shared visuals as an ordinary Isaac Lab scene asset."""
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import SpawnerCfg, get_current_stage

    def spawn_visuals(prim_path, _cfg, translation=None, orientation=None):
        stage = get_current_stage()
        prim = UsdGeom.Xform.Define(stage, prim_path)
        author_scene_visuals(stage, specs, root_path=prim_path)
        if translation is not None:
            prim.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in translation)))
        if orientation is not None:
            prim.AddOrientOp().Set(
                Gf.Quatf(float(orientation[3]), Gf.Vec3f(*(float(value) for value in orientation[:3])))
            )
        return prim.GetPrim()

    return AssetBaseCfg(prim_path=VISUAL_ROOT_PATH, spawn=SpawnerCfg(func=spawn_visuals))
