# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared finite, headless startup check for tomato and Gaussian-twin tasks."""

import argparse
from pathlib import Path

import warp as wp

wp.config.enable_backward = False
from isaaclab.app import launch_simulation


def capture_rtx_frame(visualizer):
    """Capture using the render variable's source name across OVRTX versions."""
    import numpy as np
    from ovrtx import Device

    # Headless visualizers defer rendering until capture is requested.
    try:
        visualizer.render_rgb_array()
    except RuntimeError as error:
        if "LdrColor" not in str(error):
            raise
    # The pinned Newton viewer indexes by "LdrColor", but some OVRTX releases
    # key render_vars by USD path instead. Match the stable source_name field.
    viewer = visualizer._viewer
    if viewer._render_result is not None:
        viewer._render_result.wait().fetch()
        viewer._render_result = None
    # Warm the frozen scene through renderer startup/material streaming.
    for _ in range(32):
        products = viewer._rtx.step(render_products={viewer._render_product_path}, delta_time=1 / 30)
    for product in (products or {}).values():
        for frame in reversed(product.frames):
            for variable in frame.render_vars.values():
                if variable.source_name == "LdrColor":
                    with variable.map(device=Device.CPU) as mapping:
                        return np.from_dlpack(mapping).copy()[..., :3]
    available = {
        name: [[(key, var.source_name) for key, var in frame.render_vars.items()] for frame in product.frames]
        for name, product in (products or {}).items()
    }
    raise RuntimeError(f"No LdrColor render variable available: {available}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("tomato", "gaussian_twin"), required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--capture", type=Path, help="Save the final RTX frame to a new PNG file")
    parser.add_argument("--eye", type=float, nargs=3, help="Preview camera position [m]")
    parser.add_argument("--lookat", type=float, nargs=3, help="Preview camera target [m]")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.capture and (not args.render or args.capture.exists()):
        parser.error("--capture requires --render and a new output path")
    if args.task == "tomato":
        from isaaclab_tasks.contrib.tomato_pick.env_cfg import TomatoPickEnvCfg

        cfg = TomatoPickEnvCfg()
        task = "IsaacContrib-Pick-Tomato-Franka-IK-Rel-Newton"
        device = "cpu"
    else:
        from isaaclab_tasks.contrib.stack.config.franka.stack_ik_rel_env_cfg import FrankaGaussianTwinStackNewtonEnvCfg

        cfg = FrankaGaussianTwinStackNewtonEnvCfg()
        task = "IsaacContrib-Stack-Gaussian-Twin-Franka-IK-Rel-Newton"
        device = "cuda:0"
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    if args.render:
        from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonRTXVisualizerCfg

        if cfg.sim.default_visualizer_cfg is None:
            cfg.sim.default_visualizer_cfg = NewtonRTXVisualizerCfg()
        cfg.sim.default_visualizer_cfg.headless = True
        cfg.sim.default_visualizer_cfg.window_width = 640
        cfg.sim.default_visualizer_cfg.window_height = 480
        if args.eye:
            cfg.sim.default_visualizer_cfg.eye = tuple(args.eye)
        if args.lookat:
            cfg.sim.default_visualizer_cfg.lookat = tuple(args.lookat)
    runtime = argparse.Namespace(device=device, visualizer=["newton_rtx"] if args.render else [], headless=True)
    with launch_simulation(cfg, runtime):
        import gymnasium as gym
        import torch

        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            action = torch.zeros((1, env.action_manager.total_action_dim), device=device)
            action[:, -1] = 1
            for _ in range(args.steps):
                env.step(action)
                assert torch.isfinite(env.scene["robot"].data.joint_pos.torch).all()
            if args.capture:
                from PIL import Image

                frames = [capture_rtx_frame(v) for v in env.sim.visualizers]
                frame = next((frame for frame in frames if frame is not None), None)
                if frame is None:
                    raise RuntimeError("No rendered frame available")
                args.capture.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame).save(args.capture)
                print(f"Captured {args.capture}", flush=True)
            print(f"PASS: {task}, {args.steps} steps, render={args.render}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
