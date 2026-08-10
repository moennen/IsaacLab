# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drop a USD tetrahedral mesh onto a rigid plane with Newton VBD.

The input USD must already be a standard deformable asset containing a
``UsdGeom.TetMesh``.  Use ``convert_vbd_tetmesh_usd.py`` once to convert a
custom VBD-export asset with ``vbd:vertices`` and ``vbd:tet_indices``.

Example:

    uv run python scripts/tools/simulate_vbd_tetmesh.py \
        --input-usd /path/to/soft_body.usda --visualizer newton

    uv run python scripts/tools/simulate_vbd_tetmesh.py \
        --input-usd /path/to/soft_body.usda --visualizer kit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Simulate one USD TetMesh falling onto a rigid plane with Newton VBD.")
parser.add_argument("--input-usd", type=Path, required=True, help="USD/USDZ asset containing one tetrahedral mesh.")
parser.add_argument("--drop-height", type=float, default=1.0, help="Initial height of the asset root in metres.")
parser.add_argument("--density", type=float, default=500.0, help="Deformable density in kg/m^3.")
parser.add_argument("--youngs-modulus", type=float, default=1.0e5, help="Young's modulus in Pa.")
parser.add_argument("--poissons-ratio", type=float, default=0.4, help="Poisson's ratio, strictly between 0 and 0.5.")
parser.add_argument("--particle-radius", type=float, default=0.008, help="VBD particle contact radius in metres.")
parser.add_argument("--iterations", type=int, default=10, help="VBD iterations per substep.")
parser.add_argument(
    "--max-steps", type=int, default=0, help="Stop after this many steps; 0 runs until the viewer closes."
)
add_launcher_args(parser)
parser.set_defaults(physics="newton_vbd", visualizer=["kit"])
args_cli = parser.parse_args()

if not args_cli.input_usd.is_file():
    parser.error(f"--input-usd does not exist or is not a file: {args_cli.input_usd}")
if args_cli.drop_height <= 0.0:
    parser.error("--drop-height must be positive.")
if args_cli.density <= 0.0:
    parser.error("--density must be positive.")
if args_cli.youngs_modulus <= 0.0:
    parser.error("--youngs-modulus must be positive.")
if not 0.0 < args_cli.poissons_ratio < 0.5:
    parser.error("--poissons-ratio must be strictly between 0 and 0.5.")
if args_cli.particle_radius <= 0.0:
    parser.error("--particle-radius must be positive.")
if args_cli.iterations < 1:
    parser.error("--iterations must be at least one.")
if args_cli.max_steps < 0:
    parser.error("--max-steps must be non-negative.")

visualizers = set(args_cli.visualizer or [])
if len(visualizers) != 1 or not visualizers <= {"kit", "newton"}:
    parser.error("This tool requires exactly a Kit or Newton visualizer: --visualizer kit or --visualizer newton.")

from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.physics import PhysicsCfg

from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg


def make_deformable() -> DeformableObject:
    """Spawn the USD asset and register its single TetMesh with Newton VBD."""
    nu = args_cli.poissons_ratio
    youngs_modulus = args_cli.youngs_modulus
    material = NewtonDeformableBodyMaterialCfg(
        density=args_cli.density,
        k_mu=youngs_modulus / (2.0 * (1.0 + nu)),
        k_lambda=youngs_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        # VBD treats this as a position-level contribution; a very small value
        # is robust for this deliberately minimal drop test.
        k_damp=1.0e-5,
        particle_radius=args_cli.particle_radius,
    )
    cfg = DeformableObjectCfg(
        prim_path="/World/SoftBody",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(args_cli.input_usd),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            physics_material=material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.55, 0.95)),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.drop_height)),
    )
    return DeformableObject(cfg)


def main() -> None:
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        physics_cfg.solver_cfg.iterations = args_cli.iterations
        physics_cfg.num_substeps = 4
        physics_cfg.model_cfg = NewtonModelCfg(
            soft_contact_ke=1.0e3,
            soft_contact_kd=1.0e-3,
            soft_contact_mu=0.7,
        )

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device, physics=physics_cfg)
        )
        sim.set_camera_view(eye=(2.2, 2.2, 1.8), target=(0.0, 0.0, 0.45))
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        deformable = make_deformable()
        sim.reset()
        print(f"[INFO] Simulating {args_cli.input_usd} with Newton VBD. Close the viewer to stop.")

        step = 0
        while sim.is_headless_or_exist_active_visualizer():
            # Standalone scripts do not have an environment loop to consume the
            # reset request emitted by the visualizer's "Reset Episode" button.
            # Rebuild Newton's state so the deformable particles return to the
            # initial USD pose and zero velocity.
            if sim.consume_reset_request():
                sim.reset()
                deformable.update(0.0)
                step = 0
                print("[INFO] Episode reset.")

            sim.step()
            deformable.update(sim.get_physics_dt())
            step += 1
            if args_cli.max_steps and step >= args_cli.max_steps:
                break


if __name__ == "__main__":
    main()
