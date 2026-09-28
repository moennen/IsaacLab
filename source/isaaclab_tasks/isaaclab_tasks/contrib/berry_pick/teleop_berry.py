# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the berry workcell, scripted contact validation, or continuous-grasp teleop."""

import argparse
import json
import shutil
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--berry",
    choices=["raspberry", "blackberry", "blueberry", "strawberry"],
    default="raspberry",
)
parser.add_argument(
    "--mode",
    choices=["gamepad", "keyboard", "idle", "pick", "squash"],
    default="gamepad",
)
parser.add_argument("--steps", type=int, default=0, help="0 runs until the window closes")
parser.add_argument("--no_render", action="store_true")
parser.add_argument(
    "--sync_render",
    action="store_true",
    help="Disable physics/render overlap for comparison",
)
parser.add_argument(
    "--capture_every",
    type=int,
    default=30,
    help="Image capture interval; 0 disables screenshots",
)
parser.add_argument("--window", action="store_true", help="Show the scripted benchmark in a real window")
parser.add_argument(
    "--mpm_hz",
    type=int,
    help="MPM steps/s; default: raspberry 5040, blueberry 6000, blackberry/strawberry 4560",
)
parser.add_argument("--partitions", type=int, choices=[1, 4], default=1)
parser.add_argument(
    "--render_aa",
    choices=["default", "dlaa"],
    default="default",
    help="Optional Carsten DLAA settings; default preserves the current viewer",
)
parser.add_argument(
    "--rtpt_spp",
    type=int,
    help="Explicit realtime path-tracing samples per pixel (Carsten capture: 4)",
)
parser.add_argument("--output", type=Path)
parser.add_argument(
    "--video",
    action="store_true",
    help="Record output/demo.mp4 at simulation time (adds capture/encoding cost)",
)
parser.add_argument("--gamepad_device", help="Linux joystick path, e.g. /dev/input/js0")
parser.add_argument(
    "--verify_reset",
    action="store_true",
    help="Check exact tissue/contact restoration after the run",
)
parser.add_argument(
    "--verify_render",
    action="store_true",
    help="Read back and check all native Gaussian attributes after the run",
)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
add_launcher_args(parser)
parser.set_defaults(device="cpu", visualizer=[], headless=True)
args = parser.parse_args()
if args.rtpt_spp is not None and args.rtpt_spp < 1:
    parser.error("--rtpt_spp must be positive")
if args.steps < 0 or args.width <= 0 or args.height <= 0 or args.capture_every < 0:
    parser.error("Steps/capture interval must be nonnegative and image size positive")
if args.mode == "keyboard" and args.no_render:
    parser.error("Keyboard mode requires its interactive window")
if args.verify_render and args.no_render:
    parser.error("--verify_render requires rendering")
if args.video and (args.no_render or args.output is None or shutil.which("ffmpeg") is None):
    parser.error("--video requires rendering, --output, and ffmpeg on PATH")
if args.video and (args.width % 2 or args.height % 2):
    parser.error("Video dimensions must be even")
if args.output and args.output.exists() and any(args.output.iterdir()):
    parser.error("Output directory is not empty; choose a new directory to preserve earlier runs")
if args.video and (args.output / "demo.mp4").exists():
    parser.error("Refusing to overwrite an existing demo.mp4; choose a new output directory")
if args.mode == "gamepad":
    devices = [Path(args.gamepad_device)] if args.gamepad_device else list(Path("/dev/input").glob("js*"))
    if not any(p.exists() for p in devices):
        parser.error("No Linux joystick found. Connect a controller or use --mode keyboard")
if not args.no_render:
    from isaaclab_tasks.contrib.berry_pick.render_settings import require_live_gaussian_renderer

    require_live_gaussian_renderer()

from isaaclab_tasks.contrib.berry_pick.env_cfg import BerryPickEnvCfg  # noqa: E402

cfg = BerryPickEnvCfg()
cfg.berry = args.berry
cfg.mpm_hz = args.mpm_hz
cfg.sim.device = "cpu"
with launch_simulation(cfg, args), ExitStack() as resources:
    import gymnasium as gym
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    from isaaclab_tasks.contrib.berry_pick.gamepad import BerryGamepad, BerryGamepadCfg

    env = gym.make("IsaacContrib-Pick-Berry-Franka-IK-Rel-Newton", cfg=cfg).unwrapped
    resources.callback(env.close)
    env.reset()
    viewer = None
    if not args.no_render:
        from isaaclab_tasks.contrib.berry_pick.rendering import BerryViewer

        viewer = BerryViewer(
            env,
            width=args.width,
            height=args.height,
            headless=not args.window and args.mode in ("idle", "pick", "squash"),
            pipeline=not args.sync_render,
            partitions=args.partitions,
            antialiasing=args.render_aa,
            rtpt_spp=args.rtpt_spp,
        )
        resources.callback(viewer.close)
    controller = BerryGamepad(BerryGamepadCfg(device=args.gamepad_device)) if args.mode == "gamepad" else None
    if controller is not None:
        resources.callback(controller.__exit__)
        print(controller, flush=True)
        controller.add_callback("R", lambda: (env.reset(), controller.reset()))
    action = torch.zeros((1, 7))
    action[0, 6] = 1
    rows, timings = [], []
    step = 0
    reset_held = False
    reset_verified = None
    render_verified = None
    video = None
    if args.video:
        args.output.mkdir(parents=True, exist_ok=True)
        video = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{args.width}x{args.height}",
                "-r",
                "30",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.output / "demo.mp4"),
            ],
            stdin=subprocess.PIPE,
        )
    try:
        while (not args.steps or step < args.steps) and (viewer is None or viewer.is_running()):
            started = time.perf_counter()
            if viewer is not None:
                reset_key = viewer.is_key_down("R")
                if viewer.reset_requested or (reset_key and not reset_held):
                    env.reset()
                    viewer.aperture = 0.08
                    viewer.reset_requested = False
                    if controller is not None:
                        controller.reset()
                reset_held = reset_key
                if viewer.is_paused():
                    viewer.draw(step / 30)
                    time.sleep(1 / 30)
                    continue
            if controller is not None:
                action[0] = controller.advance()
            elif args.mode in ("pick", "squash"):
                arm = env.action_manager.get_term("arm_action")
                pos, quat = arm._compute_frame_pose()
                t = step / 30
                target = np.array([0.48, 0, 0.09])
                target[2] = 0.09 - 0.0845 * np.clip((t - 1) / 3, 0, 1)
                if args.mode == "pick":
                    target[2] += 0.05 * np.clip((t - 9) / 3, 0, 1)
                current = pos[0].cpu().numpy()
                action[0, :3] = torch.from_numpy(np.clip((target - current) * 0.6, -0.008, 0.008))
                current_rot = Rotation.from_quat(quat[0].cpu().numpy())
                desired = Rotation.from_euler("x", np.pi)
                action[0, 3:6] = torch.from_numpy(np.clip((desired * current_rot.inv()).as_rotvec() * 0.2, -0.03, 0.03))
                gap = 0.013 if args.mode == "pick" else 0.001
                aperture = 0.08 + (gap - 0.08) * np.clip((t - 4) / 4, 0, 1)
                release = 14 if args.mode == "pick" else 12
                if t > release:
                    aperture += (0.08 - gap) * np.clip((t - release) / 2, 0, 1)
                action[0, 6] = aperture / 0.04 - 1
            elif args.mode == "keyboard":
                action.zero_()
                for axis, pair in enumerate(("WS", "AD", "QE", "ZX", "TG", "CV")):
                    action[0, axis] = (0.0015 if axis < 3 else 0.02) * (
                        int(viewer.is_key_down(pair[0])) - int(viewer.is_key_down(pair[1]))
                    )
                viewer.aperture = float(
                    np.clip(
                        viewer.aperture + 0.012 / 30 * (int(viewer.is_key_down("J")) - int(viewer.is_key_down("K"))),
                        0,
                        0.08,
                    )
                )
                action[0, 6] = viewer.aperture / 0.04 - 1
            env.step(action)
            simulated = time.perf_counter()
            if viewer is not None:
                viewer.draw(step / 30)
            ended = time.perf_counter()
            timings.append(
                {
                    "physics_ms": (simulated - started) * 1000,
                    "frame_ms": (ended - started) * 1000,
                }
            )
            if video is not None:
                video.stdin.write(np.ascontiguousarray(viewer.capture_image()[..., :3]).tobytes())
            if step % 30 == 0 or step == args.steps - 1:
                row = {
                    "step": step,
                    **env.berry.metrics(),
                    "aperture_command_m": float((action[0, 6] + 1) * 0.04),
                }
                row["tcp_m"] = env.action_manager.get_term("arm_action")._compute_frame_pose()[0][0].tolist()
                row["pads_local_m"] = env.berry.collider_poses(env.scene["robot"])[:, :3].tolist()
                rows.append(row)
                print(json.dumps(row), flush=True)
            if (
                viewer is not None
                and args.output
                and step > 0
                and args.capture_every
                and (step % args.capture_every == 0 or step == args.steps - 1)
            ):
                args.output.mkdir(parents=True, exist_ok=True)
                viewer.save_screenshot(str(args.output / f"frame-{step:05d}.png"))
            if args.mode in ("keyboard", "gamepad"):
                time.sleep(max(0, 1 / 30 - (time.perf_counter() - started)))
            step += 1
        if args.verify_render:
            render_verified = viewer.verify_geometry()
            print(
                "Native renderer verification passed: all dynamic arrays, opacity, and full SH3",
                flush=True,
            )
        if args.verify_reset:
            env.reset()
            for name, original in env.berry.initial.items():
                np.testing.assert_array_equal(getattr(env.berry.sim, name).numpy(), original.numpy())
            for state, original in zip(env.berry.contact.state_arrays, env.berry.contact_initial):
                np.testing.assert_array_equal(state.numpy(), original.numpy())
            np.testing.assert_allclose(
                env.action_manager.get_term("gripper_action").processed_actions.numpy(),
                0.04,
            )
            assert env.berry.tick == 0 and not env.berry.last_force.any()
            env.berry.checker.check()
            reset_verified = True
            print(
                "Reset verification passed: tissue, damage, plastic state, contact history and aperture",
                flush=True,
            )
    finally:
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            report = {
                "berry": args.berry,
                "mode": args.mode,
                "steps": step,
                "rendered": viewer is not None,
                "mpm_hz": env.berry.sim.hz,
                "mpm_cfl": float(env.berry.sim.cfl),
                "width": args.width,
                "height": args.height,
                "pipeline": not args.sync_render,
                "window": args.window or args.mode in ("gamepad", "keyboard"),
                "partitions": args.partitions,
                "render_aa": args.render_aa,
                "rtpt_spp_override": args.rtpt_spp,
                "sampling_overrides": viewer.sampling_overrides if viewer is not None else {},
                "video": args.video,
                "timing_excludes_capture_and_pacing": True,
                "reset_verified": reset_verified,
                "render_verified": render_verified,
                "rows": rows,
                "timings": timings,
                "gaussians": len(env.berry.asset["xyz"]),
                "physical_particles": len(env.berry.sim.rest),
            }
            if len(timings) > 10:
                steady = [r["frame_ms"] for r in timings[10:]]
                report["mean_frame_ms_after_warmup"] = float(np.mean(steady))
                report["p95_frame_ms_after_warmup"] = float(np.percentile(steady, 95))
                report["mean_compute_fps_after_warmup"] = 1000 / float(np.mean(steady))
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if video is not None:
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError("ffmpeg failed; the video is incomplete")
