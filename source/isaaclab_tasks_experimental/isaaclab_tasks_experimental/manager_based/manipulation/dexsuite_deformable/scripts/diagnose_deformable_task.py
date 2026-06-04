# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout diagnostics for the experimental Kuka/Allegro deformable task."""

from __future__ import annotations

import argparse
import contextlib
import time
from collections.abc import Iterable

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.dexsuite_deformable_env_cfg import (
    PhysicsCfg,
)


def _parse_env_counts(value: str) -> list[int]:
    counts = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        count = int(item)
        if count <= 0:
            raise argparse.ArgumentTypeError("environment counts must be positive")
        counts.append(count)
    if not counts:
        raise argparse.ArgumentTypeError("at least one environment count is required")
    return counts


parser = argparse.ArgumentParser(description="Diagnose deformable task physics, observations, and rollout throughput.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Dexsuite-Deformable-Kuka-Allegro-Lift-v0",
    help="Registered task name to diagnose.",
)
parser.add_argument(
    "--num-envs",
    type=_parse_env_counts,
    default=_parse_env_counts("1,16,64"),
    help="Comma-separated env counts.",
)
parser.add_argument("--steps", type=int, default=120, help="Measured rollout steps per env count.")
parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps before timing.")
parser.add_argument(
    "--action-mode",
    choices=("zero", "random"),
    default="random",
    help="Action pattern used for the rollout.",
)
parser.add_argument("--random-scale", type=float, default=0.35, help="Uniform random action scale.")
parser.add_argument(
    "--physics-preset",
    choices=("stable_two_way", "fast_two_way", "one_way_debug"),
    default="stable_two_way",
    help="Physics preset to install on the environment config.",
)
parser.add_argument("--check-interval", type=int, default=5, help="Observation health check interval in steps.")
parser.add_argument("--disable-fabric", action="store_true", default=False, help="Disable fabric scene reads/writes.")
add_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()


def _make_env_cfg(num_envs: int):
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg)
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.physics = getattr(PhysicsCfg(), args_cli.physics_preset)
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False
    return env_cfg


def _iter_tensors(value, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(item, name)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            name = f"{prefix}.{index}" if prefix else str(index)
            yield from _iter_tensors(item, name)


def _obs_health(obs) -> dict[str, float]:
    total = 0
    nonfinite = 0
    max_abs = 0.0
    for _, tensor in _iter_tensors(obs):
        tensor = tensor.detach()
        finite = torch.isfinite(tensor)
        total += tensor.numel()
        nonfinite += int((~finite).sum().item())
        if finite.any():
            max_abs = max(max_abs, float(tensor[finite].abs().max().item()))
    return {"obs_numel": total, "obs_nonfinite": nonfinite, "obs_max_abs": max_abs}


def _physics_health(env) -> dict[str, float]:
    deformable = env.scene["deformable"]
    nodal_pos = deformable.data.nodal_pos_w.torch
    nodal_vel = deformable.data.nodal_vel_w.torch
    com_env = deformable.data.root_pos_w.torch - env.scene.env_origins
    extent = nodal_pos.max(dim=1).values - nodal_pos.min(dim=1).values

    robot = env.scene["robot"]
    joint_vel = robot.data.joint_vel.torch
    joint_vel_limits = robot.data.joint_vel_limits.torch
    finite_state = (
        torch.isfinite(nodal_pos).all()
        and torch.isfinite(nodal_vel).all()
        and torch.isfinite(joint_vel).all()
    )
    joint_vel_ratio = joint_vel.abs() / joint_vel_limits.clamp_min(1.0e-6)

    try:
        command_error = env.command_manager.get_term("deformable_position").metrics["position_error"]
        command_error_mean = float(command_error.mean().item())
        command_error_max = float(command_error.max().item())
    except Exception:
        command_error_mean = float("nan")
        command_error_max = float("nan")

    return {
        "finite_state": float(bool(finite_state)),
        "com_z_min": float(com_env[:, 2].min().item()),
        "com_z_mean": float(com_env[:, 2].mean().item()),
        "max_node_vel": float(torch.linalg.norm(nodal_vel, dim=-1).max().item()),
        "max_extent": float(extent.max(dim=1).values.max().item()),
        "joint_vel_ratio_max": float(joint_vel_ratio.max().item()),
        "command_error_mean": command_error_mean,
        "command_error_max": command_error_max,
    }


def _sync_if_needed(device: str) -> None:
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _make_actions(env) -> torch.Tensor:
    action_dim = env.unwrapped.single_action_space.shape[0]
    shape = (env.unwrapped.num_envs, action_dim)
    if args_cli.action_mode == "zero":
        return torch.zeros(shape, device=env.unwrapped.device)
    return (2.0 * torch.rand(shape, device=env.unwrapped.device) - 1.0) * args_cli.random_scale


def _run_one(num_envs: int) -> dict[str, float]:
    env_cfg = _make_env_cfg(num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs_reset = env.reset()
    obs = obs_reset[0] if isinstance(obs_reset, tuple) else obs_reset

    device = env.unwrapped.device
    resets = 0
    obs_bad = 0
    obs_max_abs = 0.0
    step_times: list[float] = []

    with torch.inference_mode():
        for _ in range(args_cli.warmup_steps):
            step_out = env.step(_make_actions(env))
            obs = step_out[0]

        _sync_if_needed(device)
        for step_id in range(args_cli.steps):
            actions = _make_actions(env)
            _sync_if_needed(device)
            start = time.perf_counter()
            step_out = env.step(actions)
            _sync_if_needed(device)
            step_times.append(time.perf_counter() - start)

            obs = step_out[0]
            if len(step_out) >= 5:
                terminated, truncated = step_out[2], step_out[3]
                resets += int((terminated | truncated).sum().item())
            if args_cli.check_interval > 0 and step_id % args_cli.check_interval == 0:
                health = _obs_health(obs)
                obs_bad += health["obs_nonfinite"]
                obs_max_abs = max(obs_max_abs, health["obs_max_abs"])

    physics = _physics_health(env.unwrapped)
    env.close()

    total_time = sum(step_times)
    mean_step_ms = 1000.0 * total_time / max(len(step_times), 1)
    steps_per_s = len(step_times) / max(total_time, 1.0e-9)
    env_steps_per_s = steps_per_s * num_envs
    return {
        "num_envs": num_envs,
        "resets": resets,
        "obs_bad": obs_bad,
        "obs_max_abs": obs_max_abs,
        "mean_step_ms": mean_step_ms,
        "steps_per_s": steps_per_s,
        "env_steps_per_s": env_steps_per_s,
        **physics,
    }


def main() -> None:
    if args_cli.steps <= 0 or args_cli.warmup_steps < 0:
        raise ValueError("--steps must be positive and --warmup-steps must be non-negative")
    if args_cli.check_interval < 0:
        raise ValueError("--check-interval must be non-negative")

    torch.manual_seed(42)
    first_cfg = _make_env_cfg(args_cli.num_envs[0])
    with launch_simulation(first_cfg, args_cli):
        print(
            "envs physics action resets obs_bad max|obs| finite max_v max_extent cmd_err step_ms env_steps/s",
            flush=True,
        )
        for num_envs in args_cli.num_envs:
            result = _run_one(num_envs)
            print(
                f"{result['num_envs']:>4d} "
                f"{args_cli.physics_preset:<14s} "
                f"{args_cli.action_mode:<6s} "
                f"{result['resets']:>6d} "
                f"{result['obs_bad']:>7.0f} "
                f"{result['obs_max_abs']:>8.3g} "
                f"{result['finite_state']:>6.0f} "
                f"{result['max_node_vel']:>6.3f} "
                f"{result['max_extent']:>10.3f} "
                f"{result['command_error_mean']:>7.3f} "
                f"{result['mean_step_ms']:>7.2f} "
                f"{result['env_steps_per_s']:>11.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
