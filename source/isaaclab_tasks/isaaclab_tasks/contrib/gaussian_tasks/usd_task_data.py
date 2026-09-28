# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed USD storage for Gaussian-task metadata (no JSON or pickle payloads)."""

import numpy as np

from pxr import Sdf, Usd, Vt


def write_data(stage: Usd.Stage, path: str, value):
    """Write nested metadata as typed attributes and ordered child prims."""
    prim = stage.DefinePrim(path, "Scope")
    kind = "dict" if isinstance(value, dict) else "list" if isinstance(value, (list, tuple)) else "scalar"
    prim.CreateAttribute("data:kind", Sdf.ValueTypeNames.Token).Set(kind)
    if kind in {"dict", "list"}:
        values = list(value.values()) if kind == "dict" else value
        if kind == "dict":
            prim.CreateAttribute("data:keys", Sdf.ValueTypeNames.StringArray).Set(list(value))
        prim.CreateAttribute("data:count", Sdf.ValueTypeNames.Int).Set(len(values))
        for i, child in enumerate(values):
            write_data(stage, f"{path}/item_{i}", child)
    elif value is None:
        prim.GetAttribute("data:kind").Set("null")
    else:
        types = {
            bool: Sdf.ValueTypeNames.Bool,
            int: Sdf.ValueTypeNames.Int64,
            float: Sdf.ValueTypeNames.Double,
            str: Sdf.ValueTypeNames.String,
        }
        prim.CreateAttribute("data:value", types[type(value)]).Set(value)


def read_data(stage: Usd.Stage, path: str):
    """Read metadata written by :func:`write_data`."""
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError(f"Missing task metadata: {path}")
    kind = prim.GetAttribute("data:kind").Get()
    if kind in {"dict", "list"}:
        values = [read_data(stage, f"{path}/item_{i}") for i in range(prim.GetAttribute("data:count").Get())]
        return dict(zip(prim.GetAttribute("data:keys").Get(), values, strict=True)) if kind == "dict" else values
    if kind == "null":
        return None
    if kind != "scalar":
        raise ValueError(f"Unknown task metadata kind at {path}: {kind}")
    return prim.GetAttribute("data:value").Get()


def write_array(prim: Usd.Prim, name: str, value: np.ndarray):
    """Write a numeric array losslessly, including its shape and dtype."""
    value = np.asarray(value)
    types = {
        "f": (Sdf.ValueTypeNames.DoubleArray, Vt.DoubleArray),
        "i": (Sdf.ValueTypeNames.Int64Array, Vt.Int64Array),
        "u": (Sdf.ValueTypeNames.UInt64Array, Vt.UInt64Array),
        "b": (Sdf.ValueTypeNames.BoolArray, Vt.BoolArray),
    }
    kind, constructor = types[value.dtype.kind]
    prim.CreateAttribute(name, kind).Set(constructor(value.reshape(-1).tolist()))
    prim.CreateAttribute(name + ":shape", Sdf.ValueTypeNames.IntArray).Set(list(value.shape))
    prim.CreateAttribute(name + ":dtype", Sdf.ValueTypeNames.String).Set(value.dtype.str)


def read_array(prim: Usd.Prim, name: str) -> np.ndarray:
    """Read a numeric array written by :func:`write_array`."""
    return np.asarray(prim.GetAttribute(name).Get(), dtype=prim.GetAttribute(name + ":dtype").Get()).reshape(
        tuple(prim.GetAttribute(name + ":shape").Get())
    )
