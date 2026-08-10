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

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import warp as wp

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
    "--render-tetmesh",
    action="store_true",
    help="Render the generated TetMesh visual surface in addition to the Gaussian field.",
)
parser.add_argument(
    "--max-steps", type=int, default=0, help="Stop after this many steps; 0 runs until the viewer closes."
)
add_launcher_args(parser)
parser.set_defaults(
    physics="newton_vbd",
    visualizer=["kit"],
    kit_args=(
        "--/rtx/rtpt/gaussian/accumulatedDepth/enabled=true "
        "--/rtx/rtpt/gaussian/maxGaussiansToAccumulate=512 "
        "--/rtx/rtpt/gaussian/accumulatedAlbedo/enabled=true "
        "--/rtx/rtpt/gaussian/accumulatedDepth/allHits/enabled=true"
    ),
)
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

from isaaclab.physics import PhysicsCfg


class KeyboardGripper:
    """Two kinematic plates controlled by an SE(3) keyboard command."""

    def __init__(self, left_finger, right_finger, device: str):
        self._left_finger = left_finger
        self._right_finger = right_finger
        self._device = device
        self._initial_position = np.array((0.0, 0.0, 0.15), dtype=np.float32)
        self._initial_spacing = 0.045
        self._open_spacing = 0.22
        self.reset()

    def reset(self) -> None:
        self.position = self._initial_position.copy()
        self.rotation = Rotation.identity()
        self.spacing = self._initial_spacing

    def apply_command(self, command: torch.Tensor) -> None:
        values = command.detach().cpu().numpy()
        self.position += values[:3]
        # The keyboard rotation command is a body-frame incremental rotation.
        self.rotation = self.rotation * Rotation.from_rotvec(values[3:6])
        self.spacing = self._open_spacing if values[6] > 0.0 else self._initial_spacing

    def write_to_sim(self) -> None:
        orientation = self.rotation.as_quat().astype(np.float32)  # Isaac format: x, y, z, w.
        offset = self.rotation.apply((0.0, self.spacing * 0.5, 0.0)).astype(np.float32)
        poses = torch.tensor(
            np.array(
                [
                    np.concatenate((self.position - offset, orientation)),
                    np.concatenate((self.position + offset, orientation)),
                ],
                dtype=np.float32,
            ),
            device=self._device,
        )
        self._left_finger.write_root_link_pose_to_sim_index(root_pose=poses[0:1])
        self._right_finger.write_root_link_pose_to_sim_index(root_pose=poses[1:2])


class KitKeyboard:
    """Keyboard adapter for Kit using the same bindings as the Newton viewer."""

    def __init__(self, device: str, pos_sensitivity: float = 0.025, rot_sensitivity: float = 0.2):
        import carb
        import omni.appwindow

        self._carb = carb
        self._device = device
        self._pos_sensitivity = pos_sensitivity
        self._rot_sensitivity = rot_sensitivity
        self._held: set[str] = set()
        self._closed = True
        self._mapping = {
            "I": (0, 1.0),
            "K": (0, -1.0),
            "J": (1, 1.0),
            "L": (1, -1.0),
            "U": (2, 1.0),
            "O": (2, -1.0),
            "Z": (3, 1.0),
            "X": (3, -1.0),
            "C": (4, 1.0),
            "V": (4, -1.0),
            "B": (5, 1.0),
            "N": (5, -1.0),
        }
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def __str__(self) -> str:
        return "Keyboard: I/K J/L U/O, Z/X C/V B/N, P=open/close, R=reset"

    def reset(self) -> None:
        self._held.clear()
        self._closed = True

    def advance(self) -> torch.Tensor:
        command = np.zeros(7, dtype=np.float32)
        for key, (axis, sign) in self._mapping.items():
            if key in self._held:
                sensitivity = self._pos_sensitivity if axis < 3 else self._rot_sensitivity
                command[axis] += sign * sensitivity
        command[6] = -1.0 if self._closed else 1.0
        return torch.tensor(command, dtype=torch.float32, device=self._device)

    def _on_keyboard_event(self, event, *args) -> bool:
        name = event.input if isinstance(event.input, str) else event.input.name
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            if name == "R":
                self.reset()
            elif name == "P":
                self._closed = not self._closed
            elif name in self._mapping:
                self._held.add(name)
        elif event.type == self._carb.input.KeyboardEventType.KEY_RELEASE:
            self._held.discard(name)
        return True


class NewtonKeyboard:
    """Keyboard adapter for Newton's native OpenGL viewer."""

    def __init__(self, viewer, device: str, pos_sensitivity: float = 0.025, rot_sensitivity: float = 0.2):
        import pyglet

        self._torch_device = device
        self._pos_sensitivity = pos_sensitivity
        self._rot_sensitivity = rot_sensitivity
        self._held: set[int] = set()
        self._closed = True
        key = pyglet.window.key
        self._mapping = {
            # Newton reserves W/S/A/D/Q/E and the arrow keys for camera motion.
            key.I: (0, 1.0),
            key.K: (0, -1.0),
            key.J: (1, 1.0),
            key.L: (1, -1.0),
            key.U: (2, 1.0),
            key.O: (2, -1.0),
            key.Z: (3, 1.0),
            key.X: (3, -1.0),
            key.C: (4, 1.0),
            key.V: (4, -1.0),
            key.B: (5, 1.0),
            key.N: (5, -1.0),
        }
        self._toggle_key = key.P
        self._reset_key = key.R
        viewer.renderer.register_key_press(self._on_key_press)
        viewer.renderer.register_key_release(self._on_key_release)

    def __str__(self) -> str:
        return "Newton keyboard: I/K J/L U/O, Z/X C/V B/N, P=open/close, R=reset"

    def reset(self) -> None:
        self._held.clear()
        self._closed = True

    def advance(self) -> torch.Tensor:
        command = np.zeros(7, dtype=np.float32)
        for key, (axis, sign) in self._mapping.items():
            if key in self._held:
                sensitivity = self._pos_sensitivity if axis < 3 else self._rot_sensitivity
                command[axis] += sign * sensitivity
        command[6] = -1.0 if self._closed else 1.0
        return torch.tensor(command, dtype=torch.float32, device=self._torch_device)

    def _on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == self._reset_key:
            self.reset()
        elif symbol == self._toggle_key:
            self._closed = not self._closed
        elif symbol in self._mapping:
            self._held.add(symbol)

    def _on_key_release(self, symbol: int, modifiers: int) -> None:
        self._held.discard(symbol)


@wp.kernel(enable_backward=False)
def _skin_gaussian_positions(
    tet_positions: wp.array2d(dtype=wp.vec3f),
    rest_tet_positions: wp.array2d(dtype=wp.vec3f),
    influence_indices: wp.array(dtype=wp.int32),
    influence_weights: wp.array(dtype=wp.float32),
    rest_gaussian_positions: wp.array(dtype=wp.vec3f),
    gaussian_transforms: wp.array(dtype=wp.transformf),
):
    """Apply baked tet displacements to Gaussian centers in-place.

    ``tet_positions`` is the live Newton nodal buffer exposed by the deformable
    asset.  The other two arrays are immutable GPU buffers uploaded once from
    the USD skinning metadata.  Gaussian rotations, scales, and appearance are
    deliberately left untouched.
    """
    gaussian_id = wp.tid()
    p = rest_gaussian_positions[gaussian_id]
    base = gaussian_id * 4
    for k in range(4):
        vertex = influence_indices[base + k]
        p += (tet_positions[0, vertex] - rest_tet_positions[0, vertex]) * influence_weights[base + k]
    old = gaussian_transforms[gaussian_id]
    gaussian_transforms[gaussian_id] = wp.transformf(p, wp.transform_get_rotation(old))


@wp.kernel(enable_backward=False)
def _skin_gaussian_fabric_positions(
    fabric_positions: wp.fabricarrayarray(dtype=wp.vec3f),
    rest_tet_positions: wp.array2d(dtype=wp.vec3f),
    influence_indices: wp.array(dtype=wp.int32),
    influence_weights: wp.array(dtype=wp.float32),
    rest_gaussian_positions: wp.array(dtype=wp.vec3f),
    tet_positions: wp.array2d(dtype=wp.vec3f),
    point_count: int,
):
    """Write displacement-skinned centers in the Gaussian prim's object frame."""
    for gaussian_id in range(point_count):
        p = rest_gaussian_positions[gaussian_id]
        base = gaussian_id * 4
        for k in range(4):
            vertex = influence_indices[base + k]
            p += (tet_positions[0, vertex] - rest_tet_positions[0, vertex]) * influence_weights[base + k]
        fabric_positions[0][gaussian_id] = p


class GaussianTetSkinner:
    """GPU-only runtime bridge from Newton tet particles to Gaussian rendering."""

    def __init__(self, usd_path: Path, deformable, device: str):
        from pxr import Usd

        stage = Usd.Stage.Open(str(usd_path))
        gaussian_prim = next((p for p in stage.Traverse() if p.GetTypeName() == "ParticleField3DGaussianSplat"), None)
        tet_prim = next((p for p in stage.Traverse() if p.GetTypeName() == "TetMesh"), None)
        if gaussian_prim is None or tet_prim is None:
            raise ValueError(f"{usd_path} must contain both ParticleField3DGaussianSplat and TetMesh prims.")

        count = int(gaussian_prim.GetAttribute("newton:deformableSkin:pointCount").Get() or 0)
        influence_size = int(gaussian_prim.GetAttribute("newton:deformableSkin:influenceSize").Get() or 0)
        indices = np.asarray(gaussian_prim.GetAttribute("newton:deformableSkin:influenceIndices").Get(), dtype=np.int32)
        weights = np.asarray(gaussian_prim.GetAttribute("newton:deformableSkin:influenceWeights").Get(), dtype=np.float32)
        positions = gaussian_prim.GetAttribute("positions").Get()
        tet_positions = tet_prim.GetAttribute("points").Get()
        if count <= 0 or influence_size != 4 or len(indices) != count * 4 or len(weights) != count * 4:
            raise ValueError("Unsupported Gaussian skinning metadata: expected four influences per Gaussian.")
        if positions is None or len(positions) != count:
            raise ValueError("Gaussian skinning pointCount does not match the Gaussian positions array.")
        if tet_positions is None or len(tet_positions) == 0:
            raise ValueError("TetMesh is missing its rest points.")

        self._deformable = deformable
        self._device = device
        self._count = count
        self._indices = wp.array(indices, dtype=wp.int32, device=device)
        self._weights = wp.array(weights, dtype=wp.float32, device=device)
        # Use Newton's initialized nodal positions as the reference frame.
        # This includes the configured drop/root pose and prevents applying
        # that pose a second time to the Gaussian prim.
        self._rest_tet_positions = getattr(deformable, "_default_nodal_pos_w", None)
        if self._rest_tet_positions is None:
            raise RuntimeError("Newton did not expose initialized deformable nodal positions.")
        self._rest_gaussian_positions = wp.array(np.asarray(positions, dtype=np.float32), dtype=wp.vec3f, device=device)
        self._gaussian = None
        self._kit_selection = None

        # Newton keeps the imported Gaussian object in shape_source. Its
        # warp_data.transforms array is the renderer's GPU position buffer.
        from isaaclab_newton.physics.newton_manager import NewtonManager

        model = NewtonManager._model
        if model is not None:
            for source in model.shape_source:
                if getattr(source, "count", -1) == count and getattr(source, "warp_data", None) is not None:
                    self._gaussian = source
                    break
        if self._gaussian is None:
            raise RuntimeError("Could not locate the imported Gaussian GPU buffer in Newton's model.")
        print(f"[INFO] GPU Gaussian skinning enabled: {count} points, source={gaussian_prim.GetPath()}.")

    def _update_kit_fabric(self, tet_positions) -> None:
        """Update the Kit Gaussian prim through its Fabric GPU attribute, if present."""
        from isaaclab_newton.physics.newton_manager import NewtonManager
        try:
            import usdrt

            if self._kit_selection is None:
                self._kit_selection = NewtonManager._usdrt_stage.SelectPrims(
                    require_prim_type="ParticleField3DGaussianSplat",
                    require_attrs=[
                        (usdrt.Sdf.ValueTypeNames.Point3fArray, "positions", usdrt.Usd.Access.ReadWrite),
                    ],
                    device=self._device,
                    want_paths=True,
                )
                if self._kit_selection.GetCount() != 1:
                    print(
                        "[WARN] Expected one Gaussian Fabric positions array, "
                        f"found {self._kit_selection.GetCount()}; Kit Gaussian skinning is disabled."
                    )
                    self._kit_selection = None
                    return
            # Fabric's Gaussian renderer may refresh its attribute bundle after
            # the render delegate first discovers the prim.  Reusing the old
            # bundle can leave the renderer on the original static array.
            self._kit_selection.PrepareForReuse()
            fabric_positions = wp.fabricarrayarray(data=self._kit_selection, attrib="positions", dtype=wp.vec3f)
            wp.launch(
                _skin_gaussian_fabric_positions,
                dim=1,
                inputs=[
                    fabric_positions,
                    self._rest_tet_positions,
                    self._indices,
                    self._weights,
                    self._rest_gaussian_positions,
                    tet_positions,
                    self._count,
                ],
                device=self._device,
            )
        except Exception as exc:
            if self._kit_selection is not False:
                print(f"[WARN] Kit Gaussian Fabric update unavailable: {exc}")
                self._kit_selection = False

    def update(self) -> None:
        tet_positions = self._deformable.data.nodal_pos_w.warp
        wp.launch(
            _skin_gaussian_positions,
            dim=self._count,
            inputs=[
                tet_positions,
                self._rest_tet_positions,
                self._indices,
                self._weights,
                self._rest_gaussian_positions,
                self._gaussian.warp_data.transforms,
            ],
            device=self._device,
        )
        from isaaclab_newton.physics.newton_manager import NewtonManager

        if NewtonManager._usdrt_stage is not None and self._kit_selection is not False:
            self._update_kit_fabric(tet_positions)
        # Kit and the standalone renderer may consume the buffer from a
        # different CUDA/graphics stream.  Synchronizing here does not copy
        # data to the CPU; it only makes the GPU write visible before render.
        wp.synchronize_device(self._device)


def main() -> None:
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # Import Kit/USD-dependent modules only after SimulationApp has started.
        # Importing Newton's USD schemas before AppLauncher initializes Kit can
        # load pxr modules against an incomplete USD runtime and crash in Kit.
        import isaaclab.sim as sim_utils
        from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
        from isaaclab_newton.sim.schemas import (
            NewtonCollisionPropertiesCfg,
            NewtonDeformableBodyPropertiesCfg,
            NewtonRigidBodyPropertiesCfg,
        )
        from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

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
                    # Do not bind a material at the asset root: that would
                    # override the Gaussian field's authored SH appearance.
                    visual_material=None,
                ),
                init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.drop_height)),
            )
            return DeformableObject(cfg)

        physics_cfg.solver_cfg.iterations = args_cli.iterations
        physics_cfg.num_substeps = 4
        # The gripper pose is written every frame. CUDA graph capture would retain
        # the initial kinematic pose and make keyboard motion ineffective.
        physics_cfg.use_cuda_graph = False
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
        from pxr import UsdGeom

        tet_visual_prim = sim_utils.get_current_stage().GetPrimAtPath("/World/SoftBody/vis_mesh")
        if tet_visual_prim.IsValid():
            if args_cli.render_tetmesh:
                tet_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.55, 0.95))
                tet_material.func("/World/SoftBody/TetVisualMaterial", tet_material)
                sim_utils.bind_visual_material(
                    "/World/SoftBody/vis_mesh", "/World/SoftBody/TetVisualMaterial"
                )
            else:
                UsdGeom.Imageable(tet_visual_prim).MakeInvisible()
        else:
            print("[WARN] Generated tet visual mesh '/World/SoftBody/vis_mesh' was not found.")

        finger_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.25, 0.1))
        finger_spawn = dict(
            size=(0.30, 0.018, 0.30),
            rigid_props=NewtonRigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=NewtonCollisionPropertiesCfg(collision_enabled=True, contact_margin=0.003),
            visual_material=finger_material,
        )
        left_finger = RigidObject(
            RigidObjectCfg(
                prim_path="/World/GripperLeft",
                spawn=sim_utils.CuboidCfg(**finger_spawn),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.0225, 0.15)),
            )
        )
        right_finger = RigidObject(
            RigidObjectCfg(
                prim_path="/World/GripperRight",
                spawn=sim_utils.CuboidCfg(**finger_spawn),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0225, 0.15)),
            )
        )
        sim.reset()
        gaussian_skinner = GaussianTetSkinner(args_cli.input_usd, deformable, args_cli.device)
        deformable.update(0.0)
        gaussian_skinner.update()
        gripper = KeyboardGripper(left_finger, right_finger, args_cli.device)
        keyboard = None
        if visualizers == {"kit"}:
            keyboard = KitKeyboard(args_cli.device)
            print(keyboard)
        elif visualizers == {"newton"}:
            newton_visualizer = next(
                (visualizer for visualizer in sim.visualizers if getattr(visualizer, "_viewer", None) is not None),
                None,
            )
            if newton_visualizer is None:
                raise RuntimeError("Newton visualizer is active but its viewer is not initialized.")
            # Newton's viewer keeps Gaussian rendering disabled by default.  This
            # application uses the Gaussian field as its primary visual asset.
            # Keep Newton's regular visual channel enabled so the ground and
            # kinematic gripper plates remain visible.  Gaussian visibility is
            # controlled independently by ``show_gaussians`` below.
            newton_visualizer._viewer.show_gaussians = True
            newton_visualizer._viewer.show_visual = True
            # Deformable bodies also expose their surface triangles directly in
            # the Newton model; this is separate from the imported /vis_mesh
            # shape and must be toggled independently.
            newton_visualizer._viewer.show_triangles = args_cli.render_tetmesh
            if not args_cli.render_tetmesh:
                # Newton imports the generated TetMesh visual as a regular mesh
                # shape.  Hide only that shape; globally disabling visual shapes
                # would also hide the ground and the kinematic gripper plates.
                import newton

                model = newton_visualizer._viewer.model
                shape_labels = list(getattr(model, "shape_label", ()))
                shape_flags = np.asarray(model.shape_flags.numpy()).copy()
                visible_flag = int(newton.ShapeFlags.VISIBLE)
                hidden_tet_labels = []
                for shape_index, shape_label in enumerate(shape_labels):
                    if str(shape_label).endswith("/vis_mesh"):
                        shape_flags[shape_index] &= ~visible_flag
                        hidden_tet_labels.append(str(shape_label))
                if hidden_tet_labels:
                    model.shape_flags = wp.array(shape_flags, dtype=wp.int32, device=model.device)
                    newton_visualizer._viewer.set_model(model)
                    print(f"[INFO] Newton TetMesh visual hidden: {', '.join(hidden_tet_labels)}")
            gaussian_count = len(getattr(newton_visualizer._viewer, "_gaussian_instances", ()))
            print(f"[INFO] Newton Gaussian viewer instances: {gaussian_count}")
            keyboard = NewtonKeyboard(newton_visualizer._viewer, args_cli.device)
            print(keyboard)
        else:
            print("[INFO] Keyboard gripper control is available with --visualizer kit.")
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
                gaussian_skinner.update()
                gripper.reset()
                if keyboard is not None:
                    keyboard.reset()
                step = 0
                print("[INFO] Episode reset.")

            if keyboard is not None:
                gripper.apply_command(keyboard.advance())
                gripper.write_to_sim()
            # Render only after the GPU skinning kernel has consumed the new
            # Newton positions.  SimulationContext.step() renders immediately
            # after physics, which would otherwise render one frame before the
            # Gaussian Fabric buffer is updated.
            sim.step(render=False)
            deformable.update(sim.get_physics_dt())
            gaussian_skinner.update()
            sim.render()
            step += 1
            if args_cli.max_steps and step >= args_cli.max_steps:
                break


if __name__ == "__main__":
    main()
