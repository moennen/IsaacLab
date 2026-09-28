# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fine gamepad control for picking with the existing Linux joystick backend."""

import fcntl
import os
import struct

import numpy as np

from isaaclab.devices.device_base import DeviceCfg
from isaaclab.devices.gamepad.se3_linux_gamepad import Se3LinuxGamepad
from isaaclab.utils.configclass import configclass


class TomatoGamepad(Se3LinuxGamepad):
    """Hold L1/LB to drive; Cross/A toggles grip; Options/Menu resets."""

    def __init__(self, cfg):
        super().__init__(
            pos_sensitivity=cfg.pos_sensitivity,
            rot_sensitivity=cfg.rot_sensitivity,
            dead_zone=cfg.dead_zone,
            sim_device=cfg.sim_device,
            device=cfg.device or os.environ.get("ISAACLAB_TOMATO_GAMEPAD"),
        )
        self.connected = True
        self._buttons = {}
        self._enabled = False
        self._require_enable_release = False
        codes = bytearray(1024)
        try:
            fcntl.ioctl(self._fd, 0x84006A34, codes)  # JSIOCGBTNMAP
            self._button_codes = dict(enumerate(struct.unpack("=512H", codes)))
        except OSError:
            self._button_codes = {0: 304, 4: 310, 9: 315}

    def __str__(self):
        return (
            f"Tomato gamepad: {self._name} ({self._device_path})\n"
            "Hold L1/LB: enable motion and gripper. Left stick: X/Y. Right stick: Z/yaw.\n"
            "D-pad: roll/pitch. Cross/A: toggle grip. Options/Menu: reset.\n"
            "Release L1/LB to stop motion; the gripper keeps its current state."
        )

    def reset(self):
        super().reset()
        self._require_enable_release = any(
            self._button_codes.get(button) == 310 and held for button, held in self._buttons.items()
        )
        self._enabled = False

    def _handle_event(self, value: int, event_type: int, number: int):
        initial = bool(event_type & 0x80)
        kind = event_type & 0x7F
        if kind == 2 and number < len(self._axes):
            self._axes[number] = np.clip(value / 32767.0, -1.0, 1.0)
        elif kind == 1:
            pressed = value != 0
            was_pressed = self._buttons.get(number, False)
            self._buttons[number] = pressed
            code = self._button_codes.get(number)
            if code == 310:  # BTN_TL: L1 / LB
                if initial and pressed:
                    self._require_enable_release = True
                if not pressed:
                    self._require_enable_release = False
                self._enabled = pressed and not self._require_enable_release and not initial
            if initial or not pressed or was_pressed:
                return
            if code == 304 and self._enabled:  # BTN_SOUTH: Cross / A
                self._close_gripper = not self._close_gripper
            elif code == 315:  # BTN_START: Options / Menu
                callback = self._callbacks.get("R")
                if callback is not None:
                    callback()

    def _poll(self):
        if not self.connected:
            return
        while True:
            try:
                raw = os.read(self._fd, self._EVENT_STRUCT.size)
            except BlockingIOError:
                return
            except OSError:
                raw = b""
            if len(raw) != self._EVENT_STRUCT.size:
                self.connected = False
                self._enabled = False
                self._axes.fill(0)
                return
            _, value, kind, number = self._EVENT_STRUCT.unpack(raw)
            self._handle_event(value, kind, number)

    def advance(self):
        command = super().advance()
        if not self.connected or not self._enabled:
            command[:6] = 0
        return command

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


@configclass
class TomatoGamepadCfg(DeviceCfg):
    """Per-step command gains before the task IK action scale [m, rad]."""

    class_type: type = TomatoGamepad
    pos_sensitivity: float = 0.024
    rot_sensitivity: float = 0.08
    dead_zone: float = 0.12
    device: str | None = None
