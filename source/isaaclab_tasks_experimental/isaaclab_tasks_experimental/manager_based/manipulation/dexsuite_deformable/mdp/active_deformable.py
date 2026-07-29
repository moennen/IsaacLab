# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play-only support for showing a different deformable asset on every reset.

The heterogeneous deformable task bakes one body per environment whose tetrahedral
topology is fixed at build time, so an environment cannot hot-swap to a different
asset in place. This module keeps *all* candidate assets baked in every environment
(as separate :class:`~isaaclab_contrib.deformable.DeformableObject` bodies) and, on
each reset, randomly activates one while masking the rest.

Masking uses Newton's ``particle_flags`` ACTIVE bit: clearing it freezes a body
(the VBD solver skips it) and removes its soft contacts with the robot. Because
:meth:`DeformableObject.write_data_to_sim` restores ``particle_flags`` /
``particle_inv_mass`` from each object's ``_default_*`` snapshot every step, the
mask is written into those snapshots so it persists (the live ``model`` arrays are
written too for immediate effect).

The MDP terms all bind to a single scene entity named ``"deformable"``.
:class:`ActiveDeformableView` is injected under that name and exposes a compact,
segmented active-particle view backed by the shared flat particle arrays.

This is used only by the play variants; training is unaffected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

# NOTE: The Newton / deformable runtime stack (and its USD dependency) is imported
# lazily inside the methods below. This module is imported at config-parse time
# (the play cfg references ``select_active_deformable``), which happens before
# ``SimulationApp`` starts; importing USD here would load a second ``pxr`` and
# crash Kit at startup.

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_contrib.deformable import DeformableObject


@wp.kernel
def set_active_mask_flags(
    env_ids: wp.array(dtype=wp.int32),
    active_index: wp.array(dtype=wp.int32),
    object_index: int,
    offsets: wp.array(dtype=wp.int32),
    pristine_flags: wp.array(dtype=wp.int32),
    pristine_inv_mass: wp.array(dtype=wp.float32),
    out_default_flags: wp.array(dtype=wp.int32),
    out_default_inv_mass: wp.array(dtype=wp.float32),
    model_flags: wp.array(dtype=wp.int32),
    model_inv_mass: wp.array(dtype=wp.float32),
):
    """Activate one deformable body per env and mask the others.

    For each selected env, particle ``j`` of the object identified by
    ``object_index`` is set ACTIVE (pristine flag/inv_mass) when that object is the
    env's active choice, otherwise cleared (flag 0, inv_mass 0). Values are written
    to the object's persisted ``_default_*`` snapshot and to the live ``model``
    arrays.

    Args:
        env_ids: Environment indices being reset. Shape (num_selected,).
        active_index: Chosen active object index per env. Shape (num_envs,).
        object_index: The object this launch is writing.
        offsets: This object's per-env start index into the flat particle array.
        pristine_flags: This object's original (all-active) particle flags snapshot.
        pristine_inv_mass: This object's original inverse masses [1/kg] snapshot.
        out_default_flags: This object's ``_default_particle_flags`` to update.
        out_default_inv_mass: This object's ``_default_particle_inv_mass`` to update.
        model_flags: Live global ``model.particle_flags``.
        model_inv_mass: Live global ``model.particle_inv_mass`` [1/kg].
    """
    k, j = wp.tid()
    e = env_ids[k]
    flat = offsets[e] + j
    if active_index[e] == object_index:
        f = pristine_flags[flat]
        m = pristine_inv_mass[flat]
    else:
        f = wp.int32(0)
        m = wp.float32(0.0)
    out_default_flags[flat] = f
    out_default_inv_mass[flat] = m
    model_flags[flat] = f
    model_inv_mass[flat] = m


@wp.kernel
def gather_compact_particles_vec3f(
    src: wp.array(dtype=wp.vec3f),
    source_indices: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.vec3f),
):
    """Gather an arbitrary set of particles into a compact, one-dimensional view."""
    i = wp.tid()
    dst[i] = src[source_indices[i]]


class _TensorView:
    """Small compatibility wrapper for callers of ``*.data.root_*_w.torch``."""

    def __init__(self, tensor: torch.Tensor):
        self.torch = tensor


class ActiveDeformableData:
    """Compact active-particle data for :class:`ActiveDeformableView`.

    Unlike ``DeformableObjectData``, this deliberately does not pretend that every
    environment has the same number of nodes.  ``nodal_*`` values are flat, with
    ``node_env_ids`` and ``node_offsets`` describing their ragged segmentation.
    """

    def __init__(self, num_envs: int, device: str):
        self.device = device
        self.num_envs = num_envs
        self.node_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.node_offsets = torch.zeros(num_envs + 1, dtype=torch.long, device=device)
        self.node_env_ids = torch.empty(0, dtype=torch.long, device=device)
        self._source_indices = torch.empty(0, dtype=torch.int32, device=device)
        self._default_pos = torch.empty(0, 3, device=device)
        self._pos = torch.empty(0, 3, device=device)
        self._vel = torch.empty(0, 3, device=device)
        self._timestamp = 0
        self._gathered_timestamp = -1
        self._root_timestamp = -1
        self._root_pos = torch.zeros(num_envs, 3, device=device)
        self._root_vel = torch.zeros(num_envs, 3, device=device)

    def configure(
        self,
        source_indices: torch.Tensor,
        node_counts: torch.Tensor,
        default_pos: torch.Tensor,
    ) -> None:
        """Replace reset-time ragged metadata without padding particle buffers."""
        self.node_counts = node_counts.to(device=self.device, dtype=torch.long)
        self.node_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long, device=self.device), self.node_counts.cumsum(0))
        )
        self.node_env_ids = torch.repeat_interleave(
            torch.arange(self.num_envs, device=self.device), self.node_counts
        )
        self._source_indices = source_indices.to(device=self.device, dtype=torch.int32).contiguous()
        self._default_pos = default_pos.to(device=self.device).contiguous()
        total = int(self.node_offsets[-1].item())
        self._pos = torch.empty(total, 3, device=self.device)
        self._vel = torch.empty_like(self._pos)
        self._gathered_timestamp = -1
        self._root_timestamp = -1

    def update(self, dt: float) -> None:
        self._timestamp += 1

    def _gather(self) -> None:
        if self._gathered_timestamp == self._timestamp:
            return
        if self._source_indices.numel():
            from isaaclab_newton.physics import NewtonManager as SimulationManager

            state = SimulationManager.get_state_0()
            if state is None or state.particle_q is None or state.particle_qd is None:
                raise RuntimeError("Newton deformable particle state is unavailable.")
            indices = wp.from_torch(self._source_indices, dtype=wp.int32)
            wp.launch(
                gather_compact_particles_vec3f,
                dim=self._source_indices.numel(),
                inputs=[state.particle_q, indices],
                outputs=[wp.from_torch(self._pos, dtype=wp.vec3f)],
                device=self.device,
            )
            wp.launch(
                gather_compact_particles_vec3f,
                dim=self._source_indices.numel(),
                inputs=[state.particle_qd, indices],
                outputs=[wp.from_torch(self._vel, dtype=wp.vec3f)],
                device=self.device,
            )
        self._gathered_timestamp = self._timestamp
        self._root_timestamp = -1

    @property
    def nodal_pos_w(self) -> torch.Tensor:
        self._gather()
        return self._pos

    @property
    def nodal_vel_w(self) -> torch.Tensor:
        self._gather()
        return self._vel

    @property
    def default_nodal_pos_w(self) -> torch.Tensor:
        return self._default_pos

    def _roots(self) -> None:
        self._gather()
        if self._root_timestamp == self._timestamp:
            return
        self._root_pos.zero_()
        self._root_vel.zero_()
        if self.node_env_ids.numel():
            self._root_pos.index_add_(0, self.node_env_ids, self._pos)
            self._root_vel.index_add_(0, self.node_env_ids, self._vel)
        counts = self.node_counts.clamp_min(1).unsqueeze(1)
        self._root_pos.div_(counts)
        self._root_vel.div_(counts)
        self._root_timestamp = self._timestamp

    @property
    def root_pos_w(self) -> _TensorView:
        self._roots()
        return _TensorView(self._root_pos)

    @property
    def root_vel_w(self) -> _TensorView:
        self._roots()
        return _TensorView(self._root_vel)


def active_node_data(asset, *, velocity: bool = False, default: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flat node values, owning environment IDs, and per-env node counts.

    Normal deformables are adapted to this interface too, which keeps the MDP terms
    shared between training and the play-only heterogeneous view.
    """
    data = asset.data
    if isinstance(data, ActiveDeformableData):
        values = data.default_nodal_pos_w if default else (data.nodal_vel_w if velocity else data.nodal_pos_w)
        return values, data.node_env_ids, data.node_counts
    values_2d = data.nodal_vel_w.torch if velocity else data.nodal_pos_w.torch
    if default:
        values_2d = data.default_nodal_state_w.torch[..., :3]
    num_envs, count = values_2d.shape[:2]
    env_ids = torch.arange(num_envs, device=values_2d.device).repeat_interleave(count)
    counts = torch.full((num_envs,), count, dtype=torch.long, device=values_2d.device)
    return values_2d.reshape(-1, values_2d.shape[-1]), env_ids, counts


def segmented_min(values: torch.Tensor, env_ids: torch.Tensor, num_envs: int) -> torch.Tensor:
    out = torch.full((num_envs, values.shape[-1]), float("inf"), dtype=values.dtype, device=values.device)
    return out.scatter_reduce_(0, env_ids[:, None].expand_as(values), values, reduce="amin", include_self=True)


def segmented_max(values: torch.Tensor, env_ids: torch.Tensor, num_envs: int) -> torch.Tensor:
    out = torch.full((num_envs, values.shape[-1]), -float("inf"), dtype=values.dtype, device=values.device)
    return out.scatter_reduce_(0, env_ids[:, None].expand_as(values), values, reduce="amax", include_self=True)


def segmented_any(values: torch.Tensor, env_ids: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Segmented Boolean ``any`` compatible with CUDA's scatter-reduce support."""
    out = torch.zeros(num_envs, dtype=torch.bool, device=values.device)
    # CUDA does not implement scatter_reduce(amax) for Bool.  Assigning only
    # ``True`` entries is equivalent to an OR reduction and remains on device.
    out[env_ids[values]] = True
    return out


class ActiveDeformableView:
    """Read-only ``"deformable"`` proxy exposing the active body per environment.

    Wraps the N real single-asset deformable objects baked in every env and presents
    a :class:`DeformableObjectData` whose gathers follow a mutable active-offset
    array. It is injected into ``scene._deformable_objects["deformable"]`` so the
    existing MDP terms resolve to it unchanged. All state writes happen on the real
    objects (via :meth:`resample_active`); the proxy only reads.
    """

    def __init__(self, objects: list[DeformableObject], num_envs: int, device: str):
        if len(objects) == 0:
            raise ValueError("ActiveDeformableView requires at least one deformable object.")
        for obj in objects:
            if obj.num_instances != num_envs:
                raise ValueError("Every deformable object must span all environments.")

        self._objects = objects
        self._num_objects = len(objects)
        self._num_instances = num_envs
        self._device = device
        self._particles_per_body = [obj.max_sim_vertices_per_body for obj in objects]

        # Pristine (all-active) snapshots per object, taken before any masking.
        self._pristine_flags = [wp.clone(obj._default_particle_flags) for obj in objects]
        self._pristine_inv_mass = [wp.clone(obj._default_particle_inv_mass) for obj in objects]

        # Per-object per-env flat offsets, and the active choice per env.
        self._offset_table = torch.tensor(
            [obj._recorded_particle_offsets for obj in objects], dtype=torch.int32, device=device
        )  # (num_objects, num_envs)
        self._env_arange = torch.arange(num_envs, dtype=torch.long, device=device)
        self.active_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._asset_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Bumped whenever active_index changes, so consumers (e.g. the visualizer) can
        # cache a host copy and re-sync only on change.
        self.active_version = 0

        self._data = ActiveDeformableData(num_envs, device)
        self._refresh_compact_metadata()

    ##
    # Scene entity interface (called by InteractiveScene loops).
    ##

    @property
    def data(self) -> ActiveDeformableData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def device(self) -> str:
        return self._device

    def reset(self, env_ids=None, env_mask=None) -> None:
        """No-op; the real objects handle reset. Masking runs from the reset event."""

    def write_data_to_sim(self) -> None:
        """No-op; the underlying objects enforce their own particle state."""

    def update(self, dt: float) -> None:
        self._data.update(dt)

    ##
    # Active-asset selection.
    ##

    def apply_initial_mask(self) -> None:
        """Activate object 0 in every env before the first step (avoids frame-0 overlap).

        Poses object 0 at its exact rest (no spawn offset); the first reset applies the
        configured spawn range.
        """
        self.resample_active(self._env_arange, forced_index=0, position_range={})

    def resample_active(
        self,
        env_ids: torch.Tensor,
        forced_index: int | None = None,
        position_range: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Pick a new active body per env, mask the rest, and re-pose the active body.

        Args:
            env_ids: Environments to update. Shape (num_selected,).
            forced_index: If given, activate this object index instead of sampling.
            position_range: Per-axis uniform spawn offset [m] applied to the active
                body's rest pose (e.g. a positive ``z`` lift so it drops onto the
                table instead of clipping into it).
        """
        from isaaclab_newton.physics import NewtonManager as SimulationManager

        from isaaclab.utils.math import sample_uniform

        from isaaclab_contrib.deformable.kernels import scatter_particles_state_vec6f_mask, vec6f

        env_ids = env_ids.to(device=self._device, dtype=torch.long)
        if forced_index is None:
            choice = torch.randint(0, self._num_objects, (env_ids.shape[0],), device=self._device)
        else:
            choice = torch.full((env_ids.shape[0],), int(forced_index), dtype=torch.long, device=self._device)
        self.active_index[env_ids] = choice

        model = SimulationManager.get_model()
        env_ids_wp = wp.from_torch(env_ids.to(torch.int32).contiguous(), dtype=wp.int32)
        active_index_wp = wp.from_torch(self.active_index.to(torch.int32).contiguous(), dtype=wp.int32)

        # Mask flags/inv_mass for every object over the reset envs.
        for i, obj in enumerate(self._objects):
            wp.launch(
                set_active_mask_flags,
                dim=(env_ids.shape[0], self._particles_per_body[i]),
                inputs=[
                    env_ids_wp,
                    active_index_wp,
                    i,
                    obj._particle_offsets,
                    self._pristine_flags[i],
                    self._pristine_inv_mass[i],
                    obj._default_particle_flags,
                    obj._default_particle_inv_mass,
                    model.particle_flags,
                    model.particle_inv_mass,
                ],
                device=self._device,
            )

        # Per-env spawn offset applied to the active body's rest pose. A positive z
        # lift makes the toy drop onto the table rather than spawn clipped into it.
        ranges = position_range if position_range is not None else {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.4)}
        low = torch.tensor([ranges.get(axis, (0.0, 0.0))[0] for axis in ("x", "y", "z")], device=self._device)
        high = torch.tensor([ranges.get(axis, (0.0, 0.0))[1] for axis in ("x", "y", "z")], device=self._device)
        translation = torch.zeros(self._num_instances, 3, device=self._device)
        translation[env_ids] = sample_uniform(low, high, (env_ids.shape[0], 3), device=self._device)

        # Re-pose the active body of each reset env to its rest state (zero velocity).
        in_reset = torch.zeros(self._num_instances, dtype=torch.bool, device=self._device)
        in_reset[env_ids] = True
        for i, obj in enumerate(self._objects):
            mask = in_reset & (self.active_index == i)
            if not bool(mask.any()):
                continue
            shifted = obj.data.default_nodal_state_w.torch.clone()
            shifted[:, :, :3] += translation.unsqueeze(1)
            src_wp = wp.from_torch(shifted.contiguous(), dtype=vec6f)
            mask_wp = wp.from_torch(mask.contiguous())
            for state in obj._iter_particle_states():
                wp.launch(
                    scatter_particles_state_vec6f_mask,
                    dim=(self._num_instances, self._particles_per_body[i]),
                    inputs=[
                        src_wp,
                        mask_wp,
                        obj._particle_offsets,
                        state.particle_q,
                        state.particle_qd,
                    ],
                    device=self._device,
                )

        # Refresh compact source-index metadata. This occurs only on reset; subsequent
        # state gathers launch over exactly the selected particle total.
        self._asset_indices = self.active_index.clone()
        self._refresh_compact_metadata()
        self.active_version += 1

    def _refresh_compact_metadata(self) -> None:
        """Build selected flat indices and rest positions, once per asset selection."""
        counts_by_asset = torch.tensor(self._particles_per_body, device=self._device, dtype=torch.long)
        counts = counts_by_asset[self.active_index]
        source_offsets = self._offset_table[self.active_index, self._env_arange].to(torch.long)
        max_count = max(self._particles_per_body)
        local_ids = torch.arange(max_count, device=self._device)
        valid = local_ids.unsqueeze(0) < counts.unsqueeze(1)
        source_indices = (source_offsets.unsqueeze(1) + local_ids).masked_select(valid)

        compact_offsets = torch.cat((torch.zeros(1, dtype=torch.long, device=self._device), counts.cumsum(0)))
        default_pos = torch.empty(int(compact_offsets[-1].item()), 3, device=self._device)
        for asset_index, obj in enumerate(self._objects):
            env_ids = torch.where(self.active_index == asset_index)[0]
            if not env_ids.numel():
                continue
            count = self._particles_per_body[asset_index]
            dst = compact_offsets[env_ids].unsqueeze(1) + torch.arange(count, device=self._device)
            default_pos[dst.reshape(-1)] = obj.data.default_nodal_state_w.torch[env_ids, :count, :3].reshape(-1, 3)
        self._data.configure(source_indices, counts, default_pos)


def select_active_deformable(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]] | None = None,
    asset_name: str = "deformable",
) -> None:
    """Reset event: activate a random deformable asset per env and mask the rest.

    Ordered as a ``mode="reset"`` term; delegates to
    :meth:`ActiveDeformableView.resample_active` on the injected ``"deformable"`` proxy.

    Args:
        position_range: Per-axis uniform spawn offset [m] for the active body (a
            positive ``z`` lift makes the toy drop onto the table).
    """
    view: ActiveDeformableView = env.scene[asset_name]
    view.resample_active(env_ids, position_range=position_range)
