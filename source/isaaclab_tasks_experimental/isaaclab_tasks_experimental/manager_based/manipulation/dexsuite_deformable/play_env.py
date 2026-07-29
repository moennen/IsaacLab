# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play-only environment that swaps the active deformable asset on every reset.

The play scene bakes all candidate assets in every environment as
``deformable_0 .. deformable_{N-1}``. This env injects an
:class:`~.mdp.active_deformable.ActiveDeformableView` under the name ``"deformable"``
before the observation/reward/command managers are created, so those terms resolve
to the per-environment active body without any per-term change. Training uses the
plain :class:`~isaaclab.envs.ManagerBasedRLEnv` entry point and is unaffected.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.active_deformable import ActiveDeformableView


class DexsuiteDeformablePlayEnv(ManagerBasedRLEnv):
    """Manager-based RL env that activates one baked deformable asset per reset."""

    def load_managers(self) -> None:
        # The N single-asset bodies baked in every env are named ``deformable_<i>``.
        deformables = self.scene._deformable_objects
        names = sorted(
            (name for name in deformables if name.startswith("deformable_")),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if names:
            objects = [deformables[name] for name in names]
            view = ActiveDeformableView(objects, self.num_envs, self.device)
            # Activate object 0 everywhere before the first physics step so the
            # coincident inactive bodies are frozen + non-colliding.
            view.apply_initial_mask()
            # Inject the proxy under the name the MDP terms bind to, before the
            # managers (command/observation) cache it during load.
            deformables["deformable"] = view
        super().load_managers()
