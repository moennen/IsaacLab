# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local CLI for tomato keyboard/gamepad teleoperation and scripted picking.

Additional arguments are forwarded to the Isaac Lab teleop CLI or scripted demo.
"""

import argparse
import os
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("keyboard", "gamepad", "demo"), default="keyboard")
    parser.add_argument("--background", choices=("modeled", "splats", "0"), default=None)
    args, extra = parser.parse_known_args()
    os.environ.setdefault("ISAACLAB_NEWTON_HIDE_GROUND", "1")
    if args.background is not None:
        os.environ["ISAACLAB_TOMATO_BACKGROUND"] = args.background
    here = Path(__file__).resolve().parent
    if args.mode == "demo":
        os.environ.setdefault("ISAACLAB_TOMATO_BACKGROUND", "0")
        script = here / "scripted_pick.py"
        defaults = ["--device", "cpu"]
    else:
        script = here.parents[4] / "scripts/environments/teleoperation/teleop_se3_agent.py"
        defaults = [
            "--task",
            "IsaacContrib-Pick-Tomato-Franka-IK-Rel-Newton",
            "--num_envs",
            "1",
            "--device",
            "cpu",
            "--visualizer",
            "newton_rtx",
            "--teleop_device",
            args.mode,
            "--disable_external_cameras",
            "--no-auto_launch_cloudxr",
        ]
    sys.argv = [str(script), *defaults, *extra]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
