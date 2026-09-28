# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live MPM contact and reciprocal finger force at the robot's physics rate."""

from pathlib import Path

import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from ._mpm.explicit_mpm import ExplicitMPM
from ._mpm.hand_contact import HandContact, interpolate_pose, sticky_contact
from ._mpm.state_check import StateCheck
from .usd_asset import load_berry


@wp.kernel
def adhesive_impulse(
    attached: wp.array[int],
    force: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    dt: float,
):
    p = wp.tid()
    if attached[p] >= 0:
        wp.atomic_add(impulse, attached[p], dt * force[p])


@wp.kernel
def live_colliders(
    ticks: wp.array[int],
    start: wp.array[int],
    dt: float,
    samples: wp.array2d[wp.transform],
    poses: wp.array[wp.transform],
    following: wp.array[wp.transform],
):
    i = wp.tid()
    t = float(ticks[0] - start[0] - 1) * dt * 120.0
    poses[i] = interpolate_pose(samples, i, wp.clamp(t, 0.0, 1.0))
    following[i] = interpolate_pose(samples, i, wp.clamp(t + dt * 120.0, 0.0, 1.0))


class LiveContact(HandContact):
    """Interpolate measured finger poses over each 1/120 s contact interval."""

    def __init__(self, poses, sizes, count):
        super().__init__(
            np.stack([poses, poses]),
            np.zeros(len(poses), np.int32),
            sizes,
            count,
            rate=120,
            friction=1.2,
            strength=400,
            reach=0.005,
            lifetime=1.5,
        )
        self.start = wp.zeros(1, dtype=int)
        self.previous = poses.copy()

    def begin_step(self, sim):
        wp.launch(
            live_colliders,
            len(self.kind),
            inputs=[
                sim.ticks,
                self.start,
                sim.dt,
                self.samples,
                self.poses,
                self.following,
            ],
        )
        wp.launch(
            sticky_contact,
            len(sim.rest),
            inputs=[
                sim.x,
                sim.v,
                sim.damage,
                self.poses,
                self.kind,
                self.sizes,
                self.attached,
                self.anchor,
                self.normals,
                self.peak,
                self.age,
                self.force,
                self.strength,
                self.reach,
                self.lifetime,
                sim.spacing,
                sim.mass,
                sim.dt,
                self.particle_weight,
            ],
        )
        wp.launch(
            adhesive_impulse,
            len(sim.rest),
            inputs=[self.attached, self.force, self.impulse, sim.dt],
        )


class BerryRuntime:
    """Independent tissue state; Gaussian appearance never determines collision forces."""

    def __init__(self, cfg, robot):
        self.folder = Path(cfg.asset_root) / cfg.berry
        self.usd_path = self.folder / f"{cfg.berry}.usdz"
        self.usd_stage, self.asset, self.proxy, self.profile = load_berry(self.usd_path)
        self.offset = np.array(cfg.berry_position, np.float32)
        self.fingers, _ = robot.find_bodies("panda_(left|right)finger")
        if len(self.fingers) != 2:
            raise ValueError("Expected two Franka finger bodies")
        # The two contact boxes cover the flat inner fingertip pads [m].
        self.pad_offsets = np.array([[0, 0.0054, 0.045], [0, -0.0054, 0.045]], np.float32)
        self.sizes = np.array([[0.009, 0.0054, 0.0089]] * 2, np.float32)
        wp.set_device("cuda:0")
        poses = self.collider_poses(robot)
        self.contact = LiveContact(poses, self.sizes, len(self.proxy["xyz"]))
        params = {k: v for k, v in self.profile["simulation"].items() if k not in ("binding", "grid", "hz")}
        default_hz = 5040 if cfg.berry == "raspberry" else self.profile["simulation"]["hz"]
        # Round up the POC's 4500 Hz presets to whole 120 Hz contact intervals.
        frequency = cfg.mpm_hz if cfg.mpm_hz is not None else ((default_hz + 119) // 120) * 120
        if frequency <= 0 or frequency % 120:
            raise ValueError("MPM frequency must be a positive multiple of 120 Hz")
        params.update(
            h=self.profile["simulation"]["grid"],
            hz=frequency,
            frame_hz=120,
            cube=False,
            contact=self.contact,
            adhesion=0,
            grid_res=(121, 121, 121),
            bruise_stress=8500,
            bruise_rate=8,
        )
        # Match the accepted Allegro handling preset, distinct from the cube crush preset.
        if cfg.berry == "raspberry":
            params["bulk_yield"] = 0.8
        self.sim = ExplicitMPM(
            self.proxy["xyz"],
            regions=self.proxy["regions"],
            interface=self.proxy["interface"],
            spacing=float(self.proxy["spacing"]),
            **params,
        )
        self.sim.origin = wp.vec3(-0.12, -0.12, -0.008)
        self.sim.prepare(0)
        self.checker = StateCheck(
            self.sim.errors,
            {
                key: getattr(self.sim, key)
                for key in (
                    "x",
                    "v",
                    "c",
                    "elastic",
                    "frames",
                    "history",
                    "damage",
                    "tear",
                    "bruise_dose",
                )
            },
            capture=True,
        )
        self.initial = {
            key: wp.clone(getattr(self.sim, key))
            for key in (
                "x",
                "v",
                "c",
                "elastic",
                "frames",
                "history",
                "damage",
                "tear",
                "bruise_dose",
                "ticks",
                "clock",
            )
        }
        self.contact_initial = [wp.clone(a) for a in self.contact.state_arrays]
        self.tick = 0
        self.last_force = np.zeros((2, 3), np.float32)

    def collider_poses(self, robot):
        """Finger collision poses in the berry simulation frame [m, xyzw]."""
        q = robot.data.body_quat_w.torch[0, self.fingers].cpu().numpy()
        p = robot.data.body_pos_w.torch[0, self.fingers].cpu().numpy()
        p = p + Rotation.from_quat(q).apply(self.pad_offsets) - self.offset
        return np.column_stack([p, q]).astype(np.float32)

    def advance(self, robot):
        poses = self.collider_poses(robot)
        self.contact.samples.assign(np.stack([self.contact.previous, poses]))
        self.contact.previous = poses
        self.contact.start.assign(np.array([self.tick], np.int32))
        self.contact.impulse.zero_()
        self.sim.advance(0)
        self.tick += self.sim.hz // 120
        # Equal/opposite contact impulses feed the articulated fingers next step.
        self.last_force = -120 * self.contact.impulse.numpy()
        force = torch.as_tensor(self.last_force[None], dtype=torch.float32, device=robot.device)
        robot.set_external_force_and_torque(force, torch.zeros_like(force), body_ids=self.fingers, is_global=True)

    def reset(self, robot):
        for key, value in self.initial.items():
            getattr(self.sim, key).assign(value)
        for target, value in zip(self.contact.state_arrays, self.contact_initial):
            target.assign(value)
        self.contact.previous = self.collider_poses(robot)
        self.contact.samples.assign(np.stack([self.contact.previous] * 2))
        self.contact.start.zero_()
        self.sim.errors.zero_()
        self.tick = 0
        self.last_force.fill(0)
        zero = torch.zeros((1, 2, 3), dtype=torch.float32, device=robot.device)
        robot.set_external_force_and_torque(zero, zero, body_ids=self.fingers, is_global=True)

    def metrics(self):
        self.checker.check()
        x = self.sim.x.numpy()
        return {
            "center_m": (x.mean(0) + self.offset).tolist(),
            "height_m": float(np.ptp(x[:, 2])),
            "mean_damage": float(self.sim.damage.numpy().mean()),
            "max_damage": float(self.sim.damage.numpy().max()),
            "max_speed_m_s": float(np.linalg.norm(self.sim.v.numpy(), axis=1).max()),
            "finger_force_n": self.last_force.tolist(),
        }
