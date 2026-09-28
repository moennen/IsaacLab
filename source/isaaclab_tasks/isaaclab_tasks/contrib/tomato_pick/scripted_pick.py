# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actual IK approach/grasp/pull/release/reset probe for the registered task."""

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser()
p.add_argument(
    "--output",
    default="tomato_pick_report.json",
)
p.add_argument("--gpu-solver", action="store_true")
p.add_argument("--settle-only", action="store_true")
p.add_argument("--stop-after-close", action="store_true")
add_launcher_args(p)
p.set_defaults(device="cpu", visualizer=[], headless=True)
args = p.parse_args()
if Path(args.output).exists():
    p.error("Output already exists; choose a new --output path.")
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
from isaaclab_tasks.contrib.tomato_pick.env_cfg import TomatoPickEnvCfg

cfg = TomatoPickEnvCfg()
cfg.sim.device = args.device
cfg.sim.physics.solver_cfg.use_mujoco_cpu = not args.gpu_solver
with launch_simulation(cfg, args):
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.utils.math import compute_pose_error

    env = gym.make("IsaacContrib-Pick-Tomato-Franka-IK-Rel-Newton", cfg=cfg).unwrapped
    env.reset()
    plant = env.scene["plant"]
    ik = env.action_manager.get_term("arm_action")
    rows = []
    start = time.perf_counter()

    def move(
        label,
        target=None,
        quat=None,
        close=False,
        steps=60,
        converge=False,
        follow_fruit=False,
    ):
        stable = 0
        for i in range(600 if converge else steps):
            pos, ori = ik._compute_frame_pose()
            action = torch.zeros((1, 7), device=env.device)
            action[0, -1] = -1 if close else 1
            if follow_fruit:
                target = (
                    NewtonManager._state_0.body_q.numpy()[plant.fruit_body_ids[0], :3] + np.array([0.008, 0, 0])
                ).tolist()
            if target is not None:
                goal = torch.as_tensor(target, device=env.device, dtype=torch.float32)[None]
                rot = ori if quat is None else torch.as_tensor(quat, device=env.device, dtype=torch.float32)[None]
                dp, dr = compute_pose_error(pos, ori, goal, rot, rot_error_type="axis_angle")
                dp *= torch.clamp(
                    0.015 / torch.linalg.vector_norm(dp, dim=-1, keepdim=True).clamp_min(1e-9),
                    max=1,
                )
                dr *= torch.clamp(
                    0.10 / torch.linalg.vector_norm(dr, dim=-1, keepdim=True).clamp_min(1e-9),
                    max=1,
                )
                action[:, :3] = dp / 0.5
                action[:, 3:6] = dr / 0.5
            env.step(action)
            if label == "pull" and plant.detached[0]:
                break
            if converge:
                now, rot_now = ik._compute_frame_pose()
                ep, er = compute_pose_error(now, rot_now, goal, rot, rot_error_type="axis_angle")
                stable = stable + 1 if float(ep.norm()) < 0.003 and float(er.norm()) < 0.08 else 0
                if stable >= 10:
                    break
            if label == "pull" and i == 10:
                sd = NewtonManager._solver.mj_data
                sm = NewtonManager._solver.mj_model
                print(
                    "PULL_CONTACTS",
                    [
                        (
                            sm.geom(c.geom1).name.split("/")[-1],
                            sm.geom(c.geom2).name.split("/")[-1],
                            int(c.dim),
                            c.friction.tolist(),
                        )
                        for c in sd.contact[: sd.ncon]
                        if "tomato_00" in sm.geom(c.geom1).name + sm.geom(c.geom2).name
                    ],
                    flush=True,
                )
            if i % 15 == 0 or i == steps - 1:
                q = NewtonManager._state_0.body_q.numpy()
                ee = ik._compute_frame_pose()[0].detach().cpu().numpy()[0]
                ori_now = ik._compute_frame_pose()[1].detach().cpu().numpy()[0]
                row = dict(
                    stage=label,
                    i=i,
                    t=float(env.common_step_counter * env.step_dt),
                    ee=ee.tolist(),
                    orientation=ori_now.tolist(),
                    target=target,
                    fruit=q[plant.fruit_body_ids[0], :3].tolist(),
                    forces=plant.filtered_force.tolist(),
                    detached=plant.detached.tolist(),
                    finite=bool(np.isfinite(q).all()),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
                if not row["finite"]:
                    raise RuntimeError("Non-finite physics state")
        snapshot = Path(args.output).with_suffix("")
        snapshot.mkdir(exist_ok=True)
        np.save(snapshot / (label + ".npy"), NewtonManager._state_0.body_q.numpy())
        if label in ("enclose", "track", "close", "pull"):
            ss = NewtonManager._solver
            m = ss.mj_model
            d = ss.mj_data
            hand = next(j for j in range(m.nbody) if m.body(j).name.endswith("_panda_hand"))
            red = next(j for j in range(m.nbody) if m.body(j).name.endswith("_tomato_00"))
            contacts = []
            for j in range(d.ncon):
                c = d.contact[j]
                g1 = m.geom(c.geom1).name
                g2 = m.geom(c.geom2).name
                if "panda_" in g1 + g2:
                    f = np.zeros(6)
                    mujoco.mj_contactForce(m, d, j, f)
                    contacts.append(
                        dict(
                            g1=g1,
                            g2=g2,
                            distance=float(c.dist),
                            force=f.tolist(),
                            dim=int(c.dim),
                            friction=c.friction.tolist(),
                        )
                    )
            debug = dict(
                fruit_in_hand=(d.xmat[hand].reshape(3, 3).T @ (d.xpos[red] - d.xpos[hand])).tolist(),
                fingers=d.qpos[7:9].tolist(),
                contacts=contacts,
            )
            (snapshot / (label + "_contact.json")).write_text(json.dumps(debug, indent=2))

    # Independent native geometric Jacobian comparison at the actual TCP.
    import mujoco

    ss = NewtonManager._solver
    m = ss.mj_model
    d = ss.mj_data
    ss._update_mjc_data(d, NewtonManager.get_model(), NewtonManager._state_0)
    mujoco.mj_forward(m, d)
    hand = next(i for i in range(m.nbody) if m.body(i).name.endswith("_panda_hand"))
    tcp = d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ np.array([0, 0, 0.107])
    jp = np.zeros((3, m.nv))
    jr = jp.copy()
    mujoco.mj_jac(m, d, jp, jr, tcp, hand)
    native_jac = np.r_[jp[:, :7], jr[:, :7]]
    isaac_jac = ik._compute_frame_jacobian().detach().cpu().numpy()[0]
    jacobian_error = float(np.max(np.abs(native_jac - isaac_jac)))
    print("TCP_JACOBIAN_MAX_ERROR", jacobian_error, flush=True)
    assert jacobian_error < 1e-4, "TCP Jacobian does not match native MuJoCo"
    move("settle", steps=90)
    if not args.settle_only:
        target = NewtonManager._state_0.body_q.numpy()[plant.fruit_body_ids[0], :3].copy()
        quat = [0.5, 0.5, 0.5, 0.5]
        target = target + np.array([0.008, 0, 0])
        pre = target + np.array([-0.065, 0, 0])
        move("approach", pre.tolist(), quat, steps=120, converge=True)
        move(
            "enclose",
            target.tolist(),
            quat,
            steps=180,
            converge=True,
            follow_fruit=True,
        )
        target = NewtonManager._state_0.body_q.numpy()[plant.fruit_body_ids[0], :3].copy() + np.array([0.008, 0, 0])
        move("track", target.tolist(), quat, steps=100, converge=True, follow_fruit=True)
        move("close", target.tolist(), quat, close=True, steps=120, follow_fruit=True)
        if args.stop_after_close:
            env.close()
            raise SystemExit(0)
        target = ik._compute_frame_pose()[0].detach().cpu().numpy()[0]
        pull = target + np.array([-0.20, 0, 0.025])
        move("pull", pull.tolist(), quat, close=True, steps=150, converge=True)
        quat = [1, 0, 0, 0]
        move("carry", [0.36, -0.31, 0.28], quat, close=True, steps=120, converge=True)
        move("release", [0.36, -0.31, 0.22], quat, close=False, steps=60, converge=True)
        move("withdraw", [0.24, -0.31, 0.38], quat, close=False, steps=60, converge=True)
        move("land", steps=90)
    positions = NewtonManager._state_0.body_q.numpy()[plant.fruit_body_ids, :3]
    success = bool(
        plant.detached[0]
        and not plant.detached[1:].any()
        and 0.25 < positions[0, 0] < 0.47
        and -0.40 < positions[0, 1] < -0.22
        and 0.018 < positions[0, 2] < 0.06
    )
    result = dict(
        success=success,
        separation_thresholds_n=plant.thresholds.tolist(),
        red_attachment_active=bool(
            NewtonManager._solver.mj_data.eq_active[plant._eq_ids[0]]
            if NewtonManager._solver.use_mujoco_cpu
            else NewtonManager._solver.mjw_data.eq_active.numpy()[0, plant._eq_ids[0]]
        ),
        jacobian_max_error=jacobian_error,
        rows=rows,
        release_events=list(plant.release_events),
        elapsed_wall_s=time.perf_counter() - start,
        solver="Newton SolverMuJoCo " + ("GPU" if args.gpu_solver else "CPU"),
        fruit_positions=NewtonManager._state_0.body_q.numpy()[plant.fruit_body_ids, :3].tolist(),
    )
    env.reset()
    move("reset", steps=30)
    result["reset_detached"] = plant.detached.tolist()
    result["reset_fruit_positions"] = NewtonManager._state_0.body_q.numpy()[plant.fruit_body_ids, :3].tolist()
    Path(args.output).write_text(json.dumps(result, indent=2))
    env.close()
    assert not any(result["reset_detached"]), "Reset did not restore fruit attachments"
    if not args.settle_only:
        assert success, "Red fruit did not reach the tray without collateral separation"
