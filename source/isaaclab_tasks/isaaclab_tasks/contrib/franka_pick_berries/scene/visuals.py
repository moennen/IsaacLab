# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared USD visual scene for the berry workcell, independent of rendering backend."""

from pxr import Gf, Usd, UsdGeom, UsdShade

from ..gaussian_splats.render_delegate import author_berry_visual
from .room_scan import add_room_scan
from .tableware import add_tableware_visuals


def author_scene_visuals(
    stage: Usd.Stage, tissues, bowl_material: str = "porcelain", builtin_material: bool = True
) -> None:
    """Author berries, container, dishes and aligned room from the same assets and transforms."""
    tissues = list(tissues)
    for tissue in tissues:
        root = f"/World/Berries/{tissue.name}"
        author_berry_visual(stage, tissue, root, builtin_material)
        field = UsdGeom.Xformable(stage.GetPrimAtPath(f"{root}/Gaussians/Field"))
        field.ClearXformOpOrder()
        field.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in tissue.offset)))
    add_tableware_visuals(stage, bowl_material)
    shader = UsdShade.Shader.Get(stage, f"/World/Berries/{tissues[0].name}/Materials/Radiance/Shader")
    add_room_scan(stage, shader)
