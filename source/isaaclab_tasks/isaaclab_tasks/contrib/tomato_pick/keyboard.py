# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard input from the Newton window without importing Omniverse Kit."""

import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.sim import SimulationContext


class TomatoKeyboard(DeviceBase):
    """SE(3) increments using the familiar IsaacLab keys in a Newton window."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.viewer = next(v._viewer for v in SimulationContext.instance()._visualizers if hasattr(v, "_viewer"))
        self._callbacks = {}
        self._previous = set()
        self._closed = False
        # Reserve WASDQE for the robot. Mouse navigation remains available.
        self._gui = getattr(self.viewer, "gui", None)
        self._camera_keys = None
        if self._gui is not None:
            self._camera_keys = self._gui.update_camera_from_keys
            self._gui.update_camera_from_keys = lambda *args: None

    def __str__(self):
        return (
            "Newton tomato keyboard: W/S X, A/D Y, Q/E Z; Z/X roll, T/G pitch, C/V yaw; "
            "K gripper; R reset. Focus the render window. Mouse controls the camera."
        )

    def reset(self):
        self._closed = False
        self._previous = {key for key in set(self._callbacks) | {"K"} if self.viewer.is_key_down(key)}

    def add_callback(self, key, func):
        self._callbacks[key] = func

    def advance(self):
        if self._gui is None and getattr(self.viewer, "gui", None) is not None:
            self._gui = self.viewer.gui
            self._camera_keys = self._gui.update_camera_from_keys
            self._gui.update_camera_from_keys = lambda *args: None
        keys = set("WSADQEZXTGCVK") | set(self._callbacks)
        current = {key for key in keys if self.viewer.is_key_down(key)}
        pressed = current - self._previous
        self._previous = current
        if "K" in pressed:
            self._closed = not self._closed
        for key in pressed:
            if key in self._callbacks:
                self._callbacks[key]()
        action = torch.zeros(7, dtype=torch.float32, device=self.cfg.sim_device)
        for i, (positive, negative) in enumerate(("WS", "AD", "QE", "ZX", "TG", "CV")):
            gain = self.cfg.pos_sensitivity if i < 3 else self.cfg.rot_sensitivity
            action[i] = gain * (int(positive in current) - int(negative in current))
        action[6] = -1.0 if self._closed else 1.0
        return action

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._gui is not None and self._camera_keys is not None:
            self._gui.update_camera_from_keys = self._camera_keys
