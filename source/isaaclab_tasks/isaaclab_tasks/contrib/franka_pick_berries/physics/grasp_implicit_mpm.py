# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton's implicit MPM with a tissue damage model; the fingers hold the berry by friction.

:class:`SolverGraspImplicitMPM` extends :class:`~newton.solvers.SolverImplicitMPM` with **damage**. Newton's implicit
MPM has plasticity but no damage model. After each step, the solver accumulates a plastic strain history from the
strain rate while a particle yields, and from it damage and tearing, plus a bruise dose from sustained compression;
damage and tearing weaken the particle's shear and tensile yield limits. Stress is evaluated from the stored elastic
deformation with the tissue's elastic law.

Grasping is Newton's own frictional contact between the finger proxies and the tissue, configured as in Newton's
``example_mpm_pinch_carry`` (see :class:`GraspImplicitMPMSolverCfg`). The tissue's yield limits and damage parameters
are set per body of tissue with :meth:`SolverGraspImplicitMPM.set_tissue`.
"""

from __future__ import annotations

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp
from isaaclab_newton.physics import MPMSolver, MPMSolverCfg
from newton.solvers import SolverImplicitMPM

from isaaclab.utils import configclass

from .materials import YOUNG_MODULUS


@wp.func
def principal_stress(elastic: wp.mat33, young: float, poisson: float):
    """Principal stresses [Pa] of an elastic deformation gradient, linear elasticity in its principal stretches."""
    _u, stretch, _v = wp.svd3(elastic)
    strain = stretch - wp.vec3(1.0)
    lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    shear = young / (2.0 * (1.0 + poisson))
    return wp.vec3(lame * (strain[0] + strain[1] + strain[2])) + 2.0 * shear * strain


@wp.func
def von_mises(principal: wp.vec3):
    a = principal[0] - principal[1]
    b = principal[1] - principal[2]
    c = principal[2] - principal[0]
    return wp.sqrt(0.5 * (a * a + b * b + c * c))


@wp.struct
class DamageParameters:
    yielding_fraction: float
    strain_rate_scale: float
    onset: wp.vec2
    interval: wp.vec2
    softening: float
    tensile_ratio: float
    bruise_stress: float
    bruise_rate: float
    max_rate: float


@wp.kernel
def update_damage(
    velocity_gradient: wp.array[wp.mat33],
    elastic_strain: wp.array[wp.mat33],
    young: wp.array[float],
    poisson: wp.array[float],
    interface: wp.array[int],
    base_shear: wp.array[float],
    tear_onset: wp.array[float],
    tear_end: wp.array[float],
    dt: float,
    config: DamageParameters,
    history: wp.array[float],
    damage: wp.array[float],
    tear: wp.array[float],
    dose: wp.array[float],
    yield_stress: wp.array[float],
    tensile_ratio: wp.array[float],
):
    """Advance damage, tearing and bruising by ``dt`` [s] and weaken the particle's yield limits."""
    p = wp.tid()
    principal = principal_stress(elastic_strain[p], young[p], poisson[p])

    # Strain history, standing in for accumulated plastic strain, which Newton does not store: once a particle sits on
    # its yield surface, all further deviatoric strain is plastic, so integrate the strain rate while yielding.
    sym = 0.5 * (velocity_gradient[p] + wp.transpose(velocity_gradient[p]))
    rate = sym - (wp.trace(sym) / 3.0) * wp.identity(3, dtype=float)
    if von_mises(principal) >= config.yielding_fraction * yield_stress[p]:
        history[p] += config.strain_rate_scale * dt * wp.sqrt(2.0 / 3.0 * wp.ddot(rate, rate))

    # Bruising: a dose accumulates while the largest principal compression exceeds the threshold.
    compression = wp.max(-principal[0], wp.max(-principal[1], -principal[2]))
    excess = wp.max(compression / config.bruise_stress - 1.0, 0.0)
    dose[p] += dt * config.bruise_rate * excess * excess

    # Damage and tearing tend to their strain-history and dose targets at a bounded rate (delay damage), which keeps
    # the softening they cause from running away within a few steps.
    side = wp.where(interface[p] != 0, 1, 0)
    target = wp.clamp((history[p] - config.onset[side]) / config.interval[side], 0.0, 1.0)
    target = wp.max(target, 1.0 - wp.exp(-dose[p]))
    damage[p] = wp.max(damage[p], wp.min(target, damage[p] + config.max_rate * dt))
    if tear_end[p] > tear_onset[p]:
        failure = wp.clamp((history[p] - tear_onset[p]) / (tear_end[p] - tear_onset[p]), 0.0, 1.0)
        tear[p] = wp.max(tear[p], wp.min(failure * failure * (3.0 - 2.0 * failure), tear[p] + config.max_rate * dt))

    # Damaged tissue is weaker in shear; torn tissue loses tensile capacity but keeps supporting compression.
    yield_stress[p] = base_shear[p] * (1.0 - config.softening * damage[p])
    tensile_ratio[p] = config.tensile_ratio * (1.0 - 0.98 * tear[p])


@wp.kernel
def restore_material(
    base_shear: wp.array[float],
    tensile: float,
    yield_stress: wp.array[float],
    tensile_ratio: wp.array[float],
):
    p = wp.tid()
    yield_stress[p] = base_shear[p]
    tensile_ratio[p] = tensile


@wp.kernel
def set_range(
    start: int,
    interface: wp.array[int],
    shear: float,
    pressure: float,
    tensile: float,
    tear_onset_value: float,
    tear_end_value: float,
    interface_out: wp.array[int],
    base_shear: wp.array[float],
    tear_onset: wp.array[float],
    tear_end: wp.array[float],
    yield_stress: wp.array[float],
    yield_pressure: wp.array[float],
    tensile_ratio: wp.array[float],
):
    i = wp.tid()
    p = start + i
    interface_out[p] = interface[i]
    base_shear[p] = shear
    tear_onset[p] = tear_onset_value
    tear_end[p] = tear_end_value
    yield_stress[p] = shear
    yield_pressure[p] = pressure
    tensile_ratio[p] = tensile


@dataclass
class DamageConfig:
    """Damage model parameters shared by all tissue."""

    yielding_fraction: float = 0.9
    """Von Mises stress, relative to the current shear yield stress, above which a particle counts as yielding."""
    strain_rate_scale: float = 0.25
    """Scale of the integrated plastic strain rate, calibrated so that a full squash reaches a mean damage comparable
    to the explicit solver's."""
    onset: tuple[float, float] = (0.5, 0.08)
    """Strain history at which damage starts, in bulk tissue and at interfaces."""
    interval: tuple[float, float] = (1.0, 0.22)
    """Further history over which damage reaches 1, in bulk tissue and at interfaces."""
    softening: float = 0.25
    """Fraction of the shear yield stress that full damage removes."""
    tensile_ratio: float = 1.0
    """Tensile yield ratio of intact tissue."""
    bruise_stress: float = 4500.0
    """Principal compression [Pa] above which a bruise dose accumulates; a gentle grasp peaks near 2 kPa."""
    bruise_rate: float = 8.0
    """Bruise dose rate [1/s] at twice the threshold compression."""
    max_rate: float = 1.0
    """Largest rate [1/s] at which damage and tearing grow (delay damage, after Allix and Deu 1997). Without it, the
    softening that damage causes feeds back into more yielding within a few steps and the solve diverges."""


@dataclass
class TissueMaterial:
    """Implicit-solver yield limits and tearing of the berry tissue."""

    shear_yield: float
    """Shear yield stress [Pa]."""
    pressure_yield: float
    """Compressive yield pressure [Pa]."""
    tear_onset: float
    tear_end: float


def tissue_material(profile: dict) -> TissueMaterial:
    """Return the implicit-solver material of a berry from its asset profile.

    Compression and shear stay elastic through handling loads; tearing applies when the asset enables it.
    """
    young = YOUNG_MODULUS
    tears = profile["simulation"].get("tear_end", 0) > 0
    return TissueMaterial(
        shear_yield=0.5 * young,
        pressure_yield=5.0 * young,
        tear_onset=2.5 if tears else 0.0,
        tear_end=5.0 if tears else 0.0,
    )


# --------------------------------------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------------------------------------


class SolverGraspImplicitMPM(SolverImplicitMPM):
    """:class:`~newton.solvers.SolverImplicitMPM` with tissue damage (see the module docstring)."""

    def __init__(self, model: newton.Model, config: SolverImplicitMPM.Config, **kwargs):
        super().__init__(model, config, **kwargs)
        device = model.device
        count = model.particle_count
        self.damage_config = DamageConfig()
        """Damage model parameters; set before the first step."""
        with wp.ScopedDevice(device):
            self.interface = wp.zeros(count, dtype=int)
            self.base_shear = wp.zeros(count, dtype=float)
            self.tear_onset = wp.zeros(count, dtype=float)
            self.tear_end = wp.zeros(count, dtype=float)
            self.history = wp.zeros(count, dtype=float)
            self.damage = wp.zeros(count, dtype=float)
            self.tear = wp.zeros(count, dtype=float)
            self.dose = wp.zeros(count, dtype=float)

    def set_tissue(self, start: int, interface: np.ndarray, field: int, material: TissueMaterial) -> None:
        """Set the material of a body of tissue: the particles from ``start`` on, one per ``interface`` flag.

        Args:
            start: Index of the body's first particle in the model.
            interface: Nonzero for particles on a weak internal interface, where damage starts earlier.
            field: Unused: all tissue shares one velocity field.
            material: The body's yield limits and tearing.
        """
        del field
        mpm = self.model.mpm
        flags = wp.array(np.asarray(interface, np.int32), dtype=int, device=self.model.device)
        wp.launch(
            set_range,
            dim=len(flags),
            inputs=[
                start,
                flags,
                material.shear_yield,
                material.pressure_yield,
                self.damage_config.tensile_ratio,
                material.tear_onset,
                material.tear_end,
            ],
            outputs=[
                self.interface,
                self.base_shear,
                self.tear_onset,
                self.tear_end,
                mpm.yield_stress,
                mpm.yield_pressure,
                mpm.tensile_yield_ratio,
            ],
            device=self.model.device,
        )

    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, world_mask=world_mask, flags=flags)
        # Damage is part of the particles' history.
        for array in (self.history, self.damage, self.tear, self.dose):
            array.zero_()
        wp.launch(
            restore_material,
            dim=self.model.particle_count,
            inputs=[self.base_shear, self.damage_config.tensile_ratio],
            outputs=[self.model.mpm.yield_stress, self.model.mpm.tensile_yield_ratio],
            device=self.model.device,
        )

    def step(self, state_in, state_out, control, contacts, dt):
        model = self.model
        super().step(state_in, state_out, control, contacts, dt)
        config = self.damage_config
        parameters = DamageParameters()
        parameters.yielding_fraction = config.yielding_fraction
        parameters.strain_rate_scale = config.strain_rate_scale
        parameters.onset = wp.vec2(*config.onset)
        parameters.interval = wp.vec2(*config.interval)
        parameters.softening = config.softening
        parameters.tensile_ratio = config.tensile_ratio
        parameters.bruise_stress = config.bruise_stress
        parameters.bruise_rate = config.bruise_rate
        parameters.max_rate = config.max_rate
        # The yield limits written here are the solver's own material arrays, read directly by the next step.
        wp.launch(
            update_damage,
            dim=model.particle_count,
            inputs=[
                state_out.mpm.particle_qd_grad,
                state_out.mpm.particle_elastic_strain,
                model.mpm.young_modulus,
                model.mpm.poisson_ratio,
                self.interface,
                self.base_shear,
                self.tear_onset,
                self.tear_end,
                dt,
                parameters,
            ],
            outputs=[
                self.history,
                self.damage,
                self.tear,
                self.dose,
                model.mpm.yield_stress,
                model.mpm.tensile_yield_ratio,
            ],
            device=model.device,
        )


# --------------------------------------------------------------------------------------------------------------------
# Isaac Lab integration
# --------------------------------------------------------------------------------------------------------------------


class GraspImplicitMPMSolver(MPMSolver):
    """:class:`~isaaclab_newton.physics.MPMSolver` that builds a :class:`SolverGraspImplicitMPM`."""

    solver_class = SolverGraspImplicitMPM


@configclass
class GraspImplicitMPMSolverCfg(MPMSolverCfg):
    """Configuration of :class:`SolverGraspImplicitMPM`, with defaults tuned for berry tissue.

    The fingers hold the berry by friction, with the settings of Newton's ``example_mpm_pinch_carry``: GIMP integration,
    the residual strain of unconverged solves carried (without it, a pinched solid's grip relaxes step after step), and
    collider velocities from the fingers' successive poses (see :attr:`collider_velocity_mode`).
    """

    class_type: type = GraspImplicitMPMSolver

    grid_type: str = "sparse"
    grid_padding: int = 0
    strain_basis: str = "P1d"
    """P1d strain with many iterations resolves the elastic response; P0 or few iterations leave the tissue far softer
    than its Young's modulus."""
    transfer_scheme: str = "apic"
    integration_scheme: str = "gimp"
    carry_residual_strain: str = "unconverged"
    max_iterations: int = 100
    """Each solve is the dominant cost of a step. 50 iterations leave a carried berry visibly softer (it stretches
    out of shape)."""
    tolerance: float = 1.0e-5
    warmstart_mode: str = "auto"
    velocity_basis: str = "Q1"
    collider_basis: str = "pic27"
    collider_velocity_mode: str = "backward"
    """The coupled step hands the tissue the fingers' poses after the arm has moved them. ``"backward"`` takes the
    collider velocity from that pose and the previous one, the motion the tissue actually sees; ``"forward"``
    extrapolates from the posed velocity and lets a carried berry slip out of the fingers."""
    solver: str = "auto"
    """Newton picks the rheology solver; with GIMP integration, ``"jacobi"`` diverges on the pinched tissue."""
    separate_worlds: bool = True
