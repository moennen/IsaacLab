# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kaolin's optional, baked Simplicits representation of a berry."""

import numpy as np

from pxr import UsdGeom

COLLISION_PATH = "/Berry/TaskData/SimplicitsCollision"


def load_simplicits(stage):
    """Read Kaolin's standard skinned-physics API and the berry's MLS surface samples.

    Kaolin is imported only when this optional representation is requested. Training and
    quadrature selection happen offline; loading never trains or silently resamples a basis.
    """
    field = stage.GetPrimAtPath("/Berry/Gaussians")
    prim = stage.GetPrimAtPath(COLLISION_PATH)
    if not prim or not field.GetAttribute("kaolin_skinned_physics:default:pts").HasAuthoredValueOpinion():
        raise ValueError(
            "The berry asset has no baked Simplicits representation. Run prepare_simplicits_asset.py "
            "and select its USDZ with --berry_asset."
        )

    weights_attr = field.GetAttribute("kaolin_skinned_physics:default:renderable_skinning_weights")
    if not weights_attr or not weights_attr.HasAuthoredValueOpinion():
        raise ValueError("The Simplicits representation must include the Gaussian skinning weights")
    weights = weights_attr.Get()

    import torch
    from kaolin.physics.simplicits import SkinnedPhysicsPoints

    # Read authored API attributes directly: optional plugins can be imported after USD's
    # process-wide schema registry is initialized. Kaolin's HasAPI-based reader then misses
    # the applied API despite its authored data. Its data class still validates the buffers.
    def read(prim):
        values = {
            name: torch.as_tensor(
                np.asarray(prim.GetAttribute(f"kaolin_skinned_physics:default:{name}").Get()).copy(),
                dtype=torch.float32,
            )
            for name in ("pts", "yms", "prs", "rhos", "appx_vol", "skinning_weights", "dwdx")
        }
        count = len(values["pts"])
        values["skinning_weights"] = values["skinning_weights"].reshape(count, -1)
        values["dwdx"] = values["dwdx"].reshape(count, -1, 3)
        return SkinnedPhysicsPoints(**values)

    physics, collision = read(field), read(prim)
    result = {
        name: getattr(physics, name).detach().cpu().numpy().copy()
        for name in ("pts", "yms", "prs", "rhos", "appx_vol", "skinning_weights", "dwdx")
    }
    result["renderable_pts"] = np.asarray(field.GetAttribute("positions").Get(), np.float32).copy()
    result["renderable_weights"] = np.asarray(weights, np.float32).reshape(len(result["renderable_pts"]), -1).copy()
    result["collision"] = {
        "pts": collision.pts.numpy().copy(),
        "weights": collision.skinning_weights.numpy().copy(),
        "dwdx": collision.dwdx.numpy().copy(),
        "normals": np.asarray(UsdGeom.Points(prim).GetNormalsAttr().Get(), np.float32).copy(),
        "areas": np.asarray(prim.GetAttribute("berry:restAreas").Get(), np.float32).copy(),
    }
    handles = result["skinning_weights"].shape[1]
    if result["renderable_weights"].shape[1] != handles or result["collision"]["weights"].shape[1] != handles:
        raise ValueError("Physics, Gaussian and contact skinning must use the same handle count")
    contact = result["collision"]
    if contact["normals"].shape != contact["pts"].shape or contact["areas"].shape != (len(contact["pts"]),):
        raise ValueError("MLS normals and rest areas must match the contact samples")
    if np.any(contact["areas"] <= 0):
        raise ValueError("MLS contact samples require positive rest-area weights")
    for name in ("pts", "skinning_weights", "dwdx", "renderable_weights"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Nonfinite Simplicits data: {name}")
    return result
