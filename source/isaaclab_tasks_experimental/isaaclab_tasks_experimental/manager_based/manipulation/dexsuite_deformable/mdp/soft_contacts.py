# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Soft-contact observation and reward helpers for Newton deformable coupling."""

from __future__ import annotations

import re
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_FINGERTIP_CONTACT_BODY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("index_link_3", "index_biotac_tip"),
    ("middle_link_3", "middle_biotac_tip"),
    ("ring_link_3", "ring_biotac_tip"),
    ("thumb_link_3", "thumb_biotac_tip"),
)
"""Default Allegro per-finger body groups used for deformable contact aggregation."""


def _normalize_body_name_groups(
    body_name_groups: Sequence[str | Sequence[str]] | None,
) -> tuple[tuple[str, ...], ...]:
    """Normalize per-slot body pattern groups."""
    if body_name_groups is None:
        return DEFAULT_FINGERTIP_CONTACT_BODY_GROUPS

    normalized: list[tuple[str, ...]] = []
    for group in body_name_groups:
        if isinstance(group, str):
            normalized.append((group,))
        else:
            normalized.append(tuple(str(pattern) for pattern in group))

    if not normalized:
        raise ValueError("At least one soft-contact body group is required.")
    if any(not group for group in normalized):
        raise ValueError(f"Empty body pattern group in {body_name_groups!r}.")
    return tuple(normalized)


def _pattern_matches(pattern: str, body_name: str) -> bool:
    """Match Isaac body names with exact strings or regular expressions."""
    return body_name == pattern or re.fullmatch(pattern, body_name) is not None


class _SoftContactAggregator:
    """Cached Newton ``soft_contact_*`` aggregation for one environment and robot."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg,
        body_name_groups: tuple[tuple[str, ...], ...],
    ):
        self._env = env
        self._robot_cfg = robot_cfg
        self._body_name_groups = body_name_groups
        self._num_slots = len(body_name_groups)

        self._model_id: int | None = None
        self._body_count = -1
        self._body_to_slot: wp.array | None = None
        self._counts: wp.array | None = None
        self._counts_torch: torch.Tensor | None = None
        self._last_step = -1

    def counts(self) -> torch.Tensor:
        """Return raw active soft-contact counts shaped ``(num_envs, num_slots)``."""
        from isaaclab_newton.physics import NewtonManager

        from isaaclab_contrib.deformable.kernels import aggregate_soft_contact_counts

        model = NewtonManager._model
        contacts = NewtonManager._contacts
        if model is None or contacts is None:
            raise RuntimeError("Newton soft-contact aggregation requires an initialized Newton model and contacts.")
        has_contact_count = getattr(contacts, "soft_contact_count", None) is not None
        has_contact_shape = getattr(contacts, "soft_contact_shape", None) is not None
        if not has_contact_count or not has_contact_shape:
            raise RuntimeError("Newton contacts do not expose soft-contact buffers.")

        self._ensure_buffers(model)
        assert self._body_to_slot is not None
        assert self._counts is not None
        assert self._counts_torch is not None

        step = int(getattr(self._env, "common_step_counter", 0))
        if self._last_step == step:
            return self._counts_torch

        self._counts.zero_()
        if contacts.soft_contact_shape.shape[0] > 0:
            wp.launch(
                aggregate_soft_contact_counts,
                dim=contacts.soft_contact_shape.shape[0],
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_shape,
                    model.shape_body,
                    model.body_world,
                    self._body_to_slot,
                    self._env.num_envs,
                ],
                outputs=[self._counts],
                device=model.device,
            )

        self._last_step = step
        return self._counts_torch

    def _ensure_buffers(self, model) -> None:
        """Build body-slot maps when the Newton model changes."""
        model_id = id(model)
        body_count = int(model.body_count)
        if self._model_id == model_id and self._body_count == body_count:
            return

        labels = getattr(model, "body_label", None)
        if labels is None:
            labels = getattr(model, "body_key", None)
        if labels is None:
            labels = []
        labels = list(labels)
        if len(labels) != body_count:
            raise RuntimeError(
                f"Newton body labels are unavailable or incomplete: got {len(labels)} labels for {body_count} bodies."
            )

        body_to_slot = np.full(body_count, -1, dtype=np.int32)
        matched = [0 for _ in range(self._num_slots)]
        for body_idx, label in enumerate(labels):
            label = str(label)
            if not self._label_matches_robot_asset(label):
                continue
            body_name = label.rsplit("/", 1)[-1]
            for slot, patterns in enumerate(self._body_name_groups):
                if any(_pattern_matches(pattern, body_name) for pattern in patterns):
                    body_to_slot[body_idx] = slot
                    matched[slot] += 1
                    break

        missing = [patterns for patterns, count in zip(self._body_name_groups, matched) if count == 0]
        if missing:
            raise RuntimeError(
                "Newton model has no bodies matching soft-contact body group(s): "
                + ", ".join(str(group) for group in missing)
            )

        self._body_to_slot = wp.array(body_to_slot, dtype=wp.int32, device=model.device)
        self._counts = wp.zeros((self._env.num_envs, self._num_slots), dtype=wp.float32, device=model.device)
        self._counts_torch = wp.to_torch(self._counts)
        self._model_id = model_id
        self._body_count = body_count
        self._last_step = -1

    def _label_matches_robot_asset(self, label: str) -> bool:
        """Restrict full USD-path body labels to the selected robot asset."""
        if "/" not in label:
            return True

        robot = self._env.scene[self._robot_cfg.name]
        robot_prim_leaf = str(robot.cfg.prim_path).rstrip("/").rsplit("/", 1)[-1]
        if robot_prim_leaf.startswith("{") or not robot_prim_leaf:
            return True
        return (
            label.startswith(f"{robot_prim_leaf}/")
            or f"/{robot_prim_leaf}/" in label
            or label.endswith(f"/{robot_prim_leaf}")
        )


_SOFT_CONTACT_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _get_aggregator(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    body_name_groups: Sequence[str | Sequence[str]] | None,
) -> _SoftContactAggregator:
    """Return a cached soft-contact aggregator for the selected body groups."""
    normalized_groups = _normalize_body_name_groups(body_name_groups)
    key = (robot_cfg.name, normalized_groups)
    env_cache = _SOFT_CONTACT_CACHE.setdefault(env, {})
    aggregator = env_cache.get(key)
    if aggregator is None:
        aggregator = _SoftContactAggregator(env, robot_cfg, normalized_groups)
        env_cache[key] = aggregator
    return aggregator


def raw_fingertip_soft_contact_counts(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name_groups: Sequence[str | Sequence[str]] | None = None,
) -> torch.Tensor:
    """Raw active Newton body-particle contact counts per configured fingertip slot."""
    return _get_aggregator(env, robot_cfg, body_name_groups).counts()


def fingertip_soft_contact_counts(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name_groups: Sequence[str | Sequence[str]] | None = None,
    count_normalizer: float = 8.0,
) -> torch.Tensor:
    """Normalized soft-contact counts for policy observations."""
    counts = raw_fingertip_soft_contact_counts(env, robot_cfg, body_name_groups)
    if hasattr(env, "episode_length_buf"):
        counts = counts * (env.episode_length_buf > 0).unsqueeze(-1)
    return (counts / max(count_normalizer, 1.0e-6)).clamp(0.0, 1.0)


def fingertip_soft_contact_flags(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name_groups: Sequence[str | Sequence[str]] | None = None,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Binary soft-contact flags per configured fingertip slot."""
    counts = raw_fingertip_soft_contact_counts(env, robot_cfg, body_name_groups)
    if hasattr(env, "episode_length_buf"):
        counts = counts * (env.episode_length_buf > 0).unsqueeze(-1)
    return (counts >= contact_threshold).float()


def soft_good_contact_mask(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name_groups: Sequence[str | Sequence[str]] | None = None,
    thumb_slot: int = 3,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Return ``True`` when thumb and at least one non-thumb slot touch the deformable."""
    flags = fingertip_soft_contact_flags(env, robot_cfg, body_name_groups, contact_threshold).bool()
    num_slots = flags.shape[1]
    if thumb_slot < 0:
        thumb_slot += num_slots
    if thumb_slot < 0 or thumb_slot >= num_slots:
        raise ValueError(f"thumb_slot={thumb_slot} is outside the configured {num_slots} contact slots.")

    thumb_contact = flags[:, thumb_slot]
    non_thumb_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=flags.device)
    for slot in range(num_slots):
        if slot != thumb_slot:
            non_thumb_contact |= flags[:, slot]
    return thumb_contact & non_thumb_contact


def soft_good_contact(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name_groups: Sequence[str | Sequence[str]] | None = None,
    thumb_slot: int = 3,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward term for thumb-plus-finger deformable contact."""
    return soft_good_contact_mask(env, robot_cfg, body_name_groups, thumb_slot, contact_threshold).float()


def soft_contact_count(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name_groups: Sequence[str | Sequence[str]] | None = None,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward fraction of configured fingertip slots touching the deformable."""
    return fingertip_soft_contact_flags(env, robot_cfg, body_name_groups, contact_threshold).mean(dim=1)
