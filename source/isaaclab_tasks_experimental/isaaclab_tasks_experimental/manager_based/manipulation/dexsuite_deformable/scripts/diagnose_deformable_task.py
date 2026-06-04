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
import warp as wp

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.dexsuite_deformable_env_cfg import (
    CONTACT_BODY_GROUPS,
    FINGERTIP_LIST,
    TABLE_TOP_Z,
    PhysicsCfg,
)
from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.mdp.soft_contacts import (
    raw_fingertip_soft_contact_counts,
    soft_good_contact_mask,
)

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


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
    choices=("stable_kinematic", "fast_kinematic", "stable_two_way", "fast_two_way", "one_way_debug"),
    default="stable_kinematic",
    help="Physics preset to install on the environment config.",
)
parser.add_argument("--check-interval", type=int, default=5, help="Observation health check interval in steps.")
parser.add_argument("--disable-fabric", action="store_true", default=False, help="Disable fabric scene reads/writes.")
parser.add_argument("--disable-cuda-graph", action="store_true", default=False, help="Run Newton physics eagerly.")
parser.add_argument("--verify-cuda", action="store_true", default=False, help="Synchronize after Warp CUDA work.")
parser.add_argument("--print-warp-launches", action="store_true", default=False, help="Print each Warp kernel launch.")
parser.add_argument("--verbose-steps", action="store_true", default=False, help="Print a marker before each step.")
parser.add_argument("--print-robot-state", action="store_true", default=False, help="Print robot clearance details.")
parser.add_argument(
    "--contact-overlap-probe",
    action="store_true",
    default=False,
    help="Temporarily move the deformable onto one fingertip and print native soft-contact counts.",
)
parser.add_argument(
    "--contact-overlap-body",
    type=str,
    default="index_biotac_tip",
    help="Robot body used by the probe.",
)
parser.add_argument("--contact-overlap-steps", type=int, default=3, help="Zero-action steps after probe placement.")
add_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

if args_cli.verify_cuda:
    wp.config.verify_cuda = True
if args_cli.print_warp_launches:
    wp.config.print_launches = True


def _make_env_cfg(num_envs: int):
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg)
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.physics = getattr(PhysicsCfg(), args_cli.physics_preset)
    if args_cli.disable_cuda_graph:
        env_cfg.sim.physics.use_cuda_graph = False
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
        torch.isfinite(nodal_pos).all() and torch.isfinite(nodal_vel).all() and torch.isfinite(joint_vel).all()
    )
    joint_vel_ratio = joint_vel.abs() / joint_vel_limits.clamp_min(1.0e-6)

    try:
        command_error = env.command_manager.get_term("deformable_position").metrics["position_error"]
        command_error_mean = float(command_error.mean().item())
        command_error_max = float(command_error.max().item())
    except Exception:
        command_error_mean = float("nan")
        command_error_max = float("nan")

    try:
        soft_counts = raw_fingertip_soft_contact_counts(env, body_name_groups=CONTACT_BODY_GROUPS)
        soft_flags = (soft_counts >= 1.0).float()
        soft_good_contact = soft_good_contact_mask(env, body_name_groups=CONTACT_BODY_GROUPS)
        soft_contact_slots_mean = float(soft_flags.sum(dim=1).mean().item())
        soft_good_contact_frac = float(soft_good_contact.float().mean().item())
        soft_contact_count_max = float(soft_counts.max().item())
    except Exception:
        soft_contact_slots_mean = float("nan")
        soft_good_contact_frac = float("nan")
        soft_contact_count_max = float("nan")

    body_ids_by_name = {name: body_id for body_id, name in enumerate(robot.body_names)}
    tracked_body_names = ["palm_link", *FINGERTIP_LIST]
    tracked_body_ids = [body_ids_by_name[name] for name in tracked_body_names if name in body_ids_by_name]
    fingertip_body_ids = [body_ids_by_name[name] for name in FINGERTIP_LIST if name in body_ids_by_name]
    body_pos_env = robot.data.body_pos_w.torch - env.scene.env_origins[:, None, :]
    tracked_z = body_pos_env[:, tracked_body_ids, 2] if tracked_body_ids else torch.empty(0, device=env.device)
    fingertip_z = body_pos_env[:, fingertip_body_ids, 2] if fingertip_body_ids else torch.empty(0, device=env.device)
    palm_z = body_pos_env[:, body_ids_by_name["palm_link"], 2] if "palm_link" in body_ids_by_name else None

    joint_ids_by_name = {name: joint_id for joint_id, name in enumerate(robot.joint_names)}
    wrist_joint_pos = (
        robot.data.joint_pos.torch[:, joint_ids_by_name["iiwa7_joint_7"]]
        if "iiwa7_joint_7" in joint_ids_by_name
        else None
    )

    if args_cli.print_robot_state and env.num_envs > 0:
        first_body_z = {
            name: float(body_pos_env[0, body_id, 2].item())
            for name, body_id in zip(tracked_body_names, tracked_body_ids, strict=False)
        }
        first_joint_pos = {
            name: float(robot.data.joint_pos.torch[0, joint_id].item())
            for name, joint_id in joint_ids_by_name.items()
            if name.startswith("iiwa7_joint")
        }
        print(
            "robot_state "
            f"table_top_z={TABLE_TOP_Z:.3f} "
            f"first_body_z={first_body_z} "
            f"first_iiwa_joint_pos={first_joint_pos}",
            flush=True,
        )

    return {
        "finite_state": float(bool(finite_state)),
        "com_z_min": float(com_env[:, 2].min().item()),
        "com_z_mean": float(com_env[:, 2].mean().item()),
        "max_node_vel": float(torch.linalg.norm(nodal_vel, dim=-1).max().item()),
        "max_extent": float(extent.max(dim=1).values.max().item()),
        "joint_vel_ratio_max": float(joint_vel_ratio.max().item()),
        "tracked_min_clearance": float(tracked_z.min().item() - TABLE_TOP_Z) if tracked_z.numel() else float("nan"),
        "fingertip_min_clearance": (
            float(fingertip_z.min().item() - TABLE_TOP_Z) if fingertip_z.numel() else float("nan")
        ),
        "palm_clearance_mean": float(palm_z.mean().item() - TABLE_TOP_Z) if palm_z is not None else float("nan"),
        "wrist_joint_pos_mean": float(wrist_joint_pos.mean().item()) if wrist_joint_pos is not None else float("nan"),
        "command_error_mean": command_error_mean,
        "command_error_max": command_error_max,
        "soft_contact_slots_mean": soft_contact_slots_mean,
        "soft_good_contact_frac": soft_good_contact_frac,
        "soft_contact_count_max": soft_contact_count_max,
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


def _run_contact_overlap_probe(env) -> None:
    """Force a temporary fingertip/deformable overlap to validate soft-contact reporting."""
    if not args_cli.contact_overlap_probe:
        return

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    deformable = uenv.scene["deformable"]
    body_names = list(robot.body_names)
    body_name = args_cli.contact_overlap_body
    if body_name not in body_names:
        body_name = "index_link_3"
    body_id = body_names.index(body_name)

    tip_pos = robot.data.body_pos_w.torch[:, body_id : body_id + 1, :]
    nodal_pos = deformable.data.nodal_pos_w.torch
    offsets = nodal_pos - deformable.data.root_pos_w.torch[:, None, :]
    deformable.write_nodal_pos_to_sim_index(tip_pos + 0.2 * offsets)
    deformable.write_nodal_velocity_to_sim_index(torch.zeros_like(nodal_pos))

    zero_actions = torch.zeros(
        (uenv.num_envs, uenv.single_action_space.shape[0]),
        device=uenv.device,
    )
    for _ in range(max(args_cli.contact_overlap_steps, 1)):
        env.step(zero_actions)

    counts = raw_fingertip_soft_contact_counts(uenv, body_name_groups=CONTACT_BODY_GROUPS)
    good_contact = soft_good_contact_mask(uenv, body_name_groups=CONTACT_BODY_GROUPS)
    print(
        "contact_probe "
        f"envs={uenv.num_envs} "
        f"body={body_name} "
        f"first_counts={counts[0].detach().cpu().tolist()} "
        f"max_count={float(counts.max().item()):.0f} "
        f"good_contact_frac={float(good_contact.float().mean().item()):.3f}",
        flush=True,
    )

    env.reset()


def _run_one(num_envs: int) -> dict[str, float]:
    env_cfg = _make_env_cfg(num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs_reset = env.reset()
    obs = obs_reset[0] if isinstance(obs_reset, tuple) else obs_reset
    _run_contact_overlap_probe(env)

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
            if args_cli.verbose_steps:
                print(f"step {step_id}", flush=True)
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
            "envs physics action resets obs_bad max|obs| finite max_v max_extent robot_min_z tip_min_z "
            "palm_z wrist7 cmd_err c_slots good_c c_max step_ms env_steps/s",
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
                f"{result['tracked_min_clearance']:>11.3f} "
                f"{result['fingertip_min_clearance']:>9.3f} "
                f"{result['palm_clearance_mean']:>6.3f} "
                f"{result['wrist_joint_pos_mean']:>6.3f} "
                f"{result['command_error_mean']:>7.3f} "
                f"{result['soft_contact_slots_mean']:>7.3f} "
                f"{result['soft_good_contact_frac']:>6.3f} "
                f"{result['soft_contact_count_max']:>5.0f} "
                f"{result['mean_step_ms']:>7.2f} "
                f"{result['env_steps_per_s']:>11.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
