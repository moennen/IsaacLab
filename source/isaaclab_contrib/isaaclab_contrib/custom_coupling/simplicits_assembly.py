# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-oriented construction of a multi-world Kaolin Simplicits model.

This module intentionally imports Kaolin lazily.  The regular Isaac Lab Newton
installation must remain usable without Kaolin; only the Gaussian-twin
``--simplicits-simulation`` profile needs this experimental dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import warp as wp
from newton import Axis, ModelBuilder, ShapeFlags
from newton._src.usd import utils as usd_utils
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton.solvers import SolverMuJoCo


def _kaolin_api():
    try:
        from kaolin.experimental.newton.builder import SimplicitsModelBuilder
        from kaolin.experimental.newton.collisions import SimplicitsParticleNewtonShapeSoftContact
        from kaolin.experimental.newton.model import SimplicitsModel
        from kaolin.io import usd as kaolin_usd
        from kaolin.physics.simplicits import SimplicitsScene, SkinnedPhysicsPoints, SkinnedPoints
    except ImportError as error:
        raise ImportError(
            "Kaolin's experimental Newton/Simplicits integration is required. Install the local checkout with "
            "`uv pip install --python .venv/bin/python -e ../kaolin --no-deps`."
        ) from error
    return (
        SimplicitsModelBuilder,
        SimplicitsParticleNewtonShapeSoftContact,
        SimplicitsModel,
        SimplicitsScene,
        SkinnedPhysicsPoints,
        SkinnedPoints,
        kaolin_usd,
    )


def _find_skinned_prim(stage: Any) -> Any:
    for prim in stage.Traverse():
        if prim.GetAttribute("kaolin_skinned_physics:default:pts").IsValid():
            return prim
    raise ValueError(f"No Kaolin RKPM skinned-physics prim found in '{stage.GetRootLayer().identifier}'.")


def _load_skinned_asset(path: str, device: str) -> Any:
    """Load the RKPM-baked data authored by the companion packaging tool."""
    _, _, _, _, SkinnedPhysicsPoints, SkinnedPoints, kaolin_usd = _kaolin_api()
    from pxr import Usd

    stage = Usd.Stage.Open(path)
    if stage is None:
        raise FileNotFoundError(f"Failed to open Simplicits asset '{path}'.")
    prim = _find_skinned_prim(stage)
    skinned = kaolin_usd.get_skinned_physics(
        stage, prim.GetPath().pathString, instance_name="default", attribute="positions"
    )
    if skinned is None:
        # Kaolin schemas can be registered after USD is opened by Isaac Sim.
        # The public writer also authors these underlying attributes, so retain
        # a schema-independent reader for that startup order.
        def tensor(name: str, shape: tuple[int, ...] | None = None):
            value = prim.GetAttribute(f"kaolin_skinned_physics:default:{name}").Get()
            if value is None:
                return None
            result = torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32)
            return result.reshape(shape) if shape is not None else result

        points = tensor("pts", (-1, 3))
        appx_vol = prim.GetAttribute("kaolin_skinned_physics:default:appx_vol").Get()
        renderable_weights = tensor("renderable_skinning_weights")
        renderable_points = torch.as_tensor(
            np.asarray(prim.GetAttribute("positions").Get()).copy(), dtype=torch.float32
        )
        if points is None or appx_vol is None or renderable_weights is None:
            raise ValueError(f"'{path}' is missing required Kaolin RKPM attributes on '{prim.GetPath()}'.")
        renderable = SkinnedPoints(
            pts=renderable_points.reshape(-1, 3),
            skinning_weights=renderable_weights.reshape(renderable_points.shape[0], -1),
        )
        skinned = SkinnedPhysicsPoints(
            pts=points,
            yms=tensor("yms"),
            prs=tensor("prs"),
            rhos=tensor("rhos"),
            appx_vol=torch.tensor(float(appx_vol), dtype=torch.float32),
            skinning_weights=tensor("skinning_weights", (points.shape[0], -1)),
            dwdx=tensor("dwdx", (points.shape[0], -1, 3)),
            renderable=renderable,
        )
    return skinned.to(device=device, dtype=torch.float32)


def _subsample_preserving_renderable(skinned: Any, maximum: int | None) -> Any:
    if maximum is None or maximum <= 0 or maximum >= int(skinned.pts.shape[0]):
        return skinned
    sampled = skinned.subsample(num_pts=int(maximum))
    # Kaolin's generic subsample intentionally only carries simulation data.
    # Gaussian points are rendered through the original handle weights.
    sampled.renderable = skinned.renderable
    return sampled


def _register_mujoco_attributes(builder: ModelBuilder) -> None:
    SolverMuJoCo.register_custom_attributes(builder)


def _build_rigid_proto(stage: Any, env_path: str, up_axis: str) -> ModelBuilder:
    """Import the stack scene's rigid content while excluding every twin slot."""
    builder = ModelBuilder(up_axis=up_axis)
    _register_mujoco_attributes(builder)
    env_prim = stage.GetPrimAtPath(env_path)
    env_to_local = wp.transform_inverse(usd_utils.get_transform(env_prim, local=False))
    resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    for child in env_prim.GetChildren():
        if child.GetName().startswith("GaussianTwin_"):
            continue
        builder.add_usd(
            stage,
            root_path=child.GetPath().pathString,
            xform=env_to_local,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            schema_resolvers=resolvers,
        )

    # The stack task's authored ground is global and sits at z=-1.05 m, below
    # the SeattleLabTable.  Do not substitute ``add_ground_plane()`` here: its
    # default z=0 plane cuts through the table top, making the robot and twin
    # appear to be placed on the floor.  Import the actual task prim into the
    # per-world prototype so the Simplicits path retains the same layout as
    # VBD and rigid simulation.
    ground = stage.GetPrimAtPath("/World/GroundPlane")
    if ground and ground.IsValid():
        builder.add_usd(
            stage,
            root_path=ground.GetPath().pathString,
            xform=env_to_local,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            schema_resolvers=resolvers,
        )
    for shape_id in range(builder.shape_count):
        builder.shape_flags[shape_id] |= ShapeFlags.COLLIDE_PARTICLES
    return builder


def _build_gaussian_proto(stage: Any, env_path: str, slot: int, up_axis: str) -> ModelBuilder:
    """Import only the field shape, never its authored TetMesh proxy."""
    builder = ModelBuilder(up_axis=up_axis)
    _register_mujoco_attributes(builder)
    env_prim = stage.GetPrimAtPath(env_path)
    from pxr import Usd

    slot_prim = stage.GetPrimAtPath(f"{env_path}/GaussianTwin_{slot}")
    gaussian = next(
        (prim for prim in Usd.PrimRange(slot_prim) if prim.GetTypeName() == "ParticleField3DGaussianSplat"),
        None,
    )
    if gaussian is None:
        raise ValueError(f"GaussianTwin_{slot} has no ParticleField3DGaussianSplat prim.")
    builder.add_usd(
        stage,
        root_path=gaussian.GetPath().pathString,
        xform=wp.transform_inverse(usd_utils.get_transform(env_prim, local=False)),
        load_visual_shapes=True,
        skip_mesh_approximation=True,
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
    )
    return builder


def _transform(
    env_position: tuple[float, float, float], slot_position: tuple[float, float, float], device: str
) -> torch.Tensor:
    result = torch.eye(4, device=device, dtype=torch.float32)
    result[:3, 3] = torch.as_tensor(env_position, device=device) + torch.as_tensor(slot_position, device=device)
    return result


def build_gaussian_twin_simplicits_model(
    *,
    stage: Any,
    env_paths: list[str],
    env_positions: list[tuple[float, float, float]],
    asset_paths: list[str],
    slot_positions: list[tuple[float, float, float]],
    device: str,
    up_axis: str,
    gravity: float,
    max_quadrature_points: int | None,
    num_newton_steps: int,
    cg_iterations: int,
    cg_tolerance: float,
    newton_convergence_tolerance: float,
    particle_radius: float,
    soft_contact_ke: float,
    soft_contact_mu: float,
    soft_contact_coefficient: float,
    enable_inter_object_collisions: bool,
    pre_finalize_callback: Callable[[Any], None] | None = None,
) -> tuple[Any, list[list[int]], list[list[tuple[int, int]]]]:
    """Assemble robot worlds and RKPM objects in deterministic env/slot order."""
    if len(asset_paths) != len(slot_positions):
        raise ValueError("Every Simplicits asset needs one slot position.")
    if not env_paths:
        raise ValueError("Cannot construct a Simplicits model without environments.")

    (
        SimplicitsModelBuilder,
        SimplicitsParticleNewtonShapeSoftContact,
        SimplicitsModel,
        SimplicitsScene,
        _SkinnedPhysicsPoints,
        _SkinnedPoints,
        _kaolin_usd,
    ) = _kaolin_api()
    assets = [_load_skinned_asset(path, device) for path in asset_paths]
    base = SimplicitsModelBuilder(up_axis=Axis.from_string(up_axis), gravity=-float(gravity))
    model = SimplicitsModel(device)
    model.simplicits_scene = SimplicitsScene(device=device)
    scene = model.simplicits_scene
    scene.max_newton_steps = int(num_newton_steps)
    scene.cg_iters = int(cg_iterations)
    # Kaolin's generic 1e-4 tolerances are much larger than the reduced
    # residuals generated by a centimetre-scale toy in SI units.  They make
    # CG return the zero direction and consequently freeze the object even
    # while the collision pipeline reports contacts.
    scene.cg_tol = float(cg_tolerance)
    scene.conv_tol = float(newton_convergence_tolerance)
    scene.direct_solve = False

    object_ids: list[list[int]] = []
    initial_transforms: list[list[torch.Tensor]] = []
    for env_position in env_positions:
        env_ids: list[int] = []
        env_transforms: list[torch.Tensor] = []
        for slot, asset in enumerate(assets):
            initial_transform = _transform(env_position, slot_positions[slot], device)
            env_ids.append(
                scene.add_object(
                    _subsample_preserving_renderable(asset, max_quadrature_points),
                    num_qp=None,
                    init_transform=initial_transform,
                    is_kinematic=False,
                    normalize_weights_by_samples=True,
                    apply_qr=True,
                )
            )
            env_transforms.append(initial_transform.clone())
        object_ids.append(env_ids)
        initial_transforms.append(env_transforms)
    if enable_inter_object_collisions:
        # Simplicits owns reduced soft/soft contact.  A single toy needs only
        # the Newton rigid/soft contact term, while concurrent toys need this
        # Kaolin-native pair force; NewtonCollisionPipelineCfg cannot express
        # RKPM object pairs.
        scene.enable_collisions(
            collision_particle_radius=float(particle_radius),
            detection_ratio=1.5,
            impenetrable_barrier_ratio=0.25,
            collision_penalty=float(soft_contact_ke),
            max_contact_pairs=10000,
            friction=float(soft_contact_mu),
        )
    # Kaolin's ``Gravity`` term uses ``g dot x`` as potential energy and
    # minimizes it.  Its ``g`` is therefore the *opposite* of the physical
    # acceleration vector (the default Y-up example likewise passes +9.8 for
    # downward gravity).  This task is Z-up, so pass +|g| to obtain a physical
    # -Z acceleration. Passing -9.81 here launches the toy upward.
    scene.set_scene_gravity(torch.tensor((0.0, 0.0, float(gravity)), device=device, dtype=torch.float32))

    rigid_proto = _build_rigid_proto(stage, env_paths[0], up_axis)
    gaussian_protos = [_build_gaussian_proto(stage, env_paths[0], slot, up_axis) for slot in range(len(assets))]
    sim_points = scene.sim_pts.numpy()
    sim_masses = scene.sim_M.values.numpy()[::3]
    offsets = [0]
    ordered_object_ids = [object_id for env_ids in object_ids for object_id in env_ids]
    for object_id in ordered_object_ids:
        offsets.append(offsets[-1] + scene.sim_obj_dict[object_id].num_qp)

    particle_ranges: list[list[tuple[int, int]]] = []
    base.model = model
    start_particle = len(base.particle_q)
    label_attributes = ("articulation_label", "body_label", "joint_label", "shape_label")
    for env_index, env_path in enumerate(env_paths):
        base.begin_world()
        label_starts = {name: len(getattr(base, name)) for name in label_attributes}
        base.add_builder(rigid_proto, xform=wp.transform(env_positions[env_index], wp.quat_identity()))
        for slot, gaussian_proto in enumerate(gaussian_protos):
            base.add_builder(gaussian_proto, xform=wp.transform(env_positions[env_index], wp.quat_identity()))
        if env_index:
            for name in label_attributes:
                labels = getattr(base, name)
                for index in range(label_starts[name], len(labels)):
                    labels[index] = labels[index].replace(env_paths[0], env_path, 1)

        local_ranges: list[tuple[int, int]] = []
        for slot in range(len(assets)):
            object_index = env_index * len(assets) + slot
            begin, end = offsets[object_index], offsets[object_index + 1]
            particle_start = len(base.particle_q)
            base.add_particles(
                pos=[tuple(float(value) for value in point) for point in sim_points[begin:end]],
                vel=[(0.0, 0.0, 0.0)] * (end - begin),
                mass=[float(value) for value in sim_masses[begin:end]],
                radius=[float(particle_radius)] * (end - begin),
            )
            local_ranges.append((particle_start, len(base.particle_q)))
        base.end_world()
        particle_ranges.append(local_ranges)
    end_particle = len(base.particle_q)
    # Isaac Lab articulation/sensor MODEL_INIT callbacks configure the builder
    # and may append frame-transform sites. They must run before our manual
    # finalize below, rather than after the facade hands the finished model to
    # NewtonManager.
    if pre_finalize_callback is not None:
        pre_finalize_callback(base)

    # Bypass SimplicitsModelBuilder.finalize(), which appends one global
    # particle batch and therefore cannot retain Isaac Lab's world layout.
    base_model = base.__class__.__mro__[1].finalize(base, device=device, requires_grad=False)
    model.__dict__.update(base_model.__dict__)
    model.simplicits_particle_start = start_particle
    model.simplicits_particle_end = end_particle
    model._simplicits_object_ids = object_ids
    model._simplicits_particle_ranges = particle_ranges
    model._simplicits_initial_transforms = initial_transforms
    model.soft_contact_ke = float(soft_contact_ke)
    model.soft_contact_mu = float(soft_contact_mu)
    scene.force_dict["pt_wise"]["newton_soft_collisions"] = {
        "object": SimplicitsParticleNewtonShapeSoftContact(
            model, wp.ones_like(scene.sim_vols), dt=scene.timestep, friction_use_lagged_body_contact_force_norm=False
        ),
        "coeff": float(soft_contact_coefficient),
    }
    return model, object_ids, particle_ranges
