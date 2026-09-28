# Tomato picking

Follow [runtime installation and Nucleus download](../gaussian_tasks/README.md)
first. Task source and its CLI live here; no external tomato authoring checkout
is required. Assets default to the shared local task cache;
`ISAACLAB_TOMATO_ASSET_ROOT` optionally overrides the tomato folder.

From the IsaacLab root:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-tasks"
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/tomato_pick/teleop_tomato.py --mode keyboard
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/tomato_pick/teleop_tomato.py --mode gamepad
```

The wrapper invokes the branch's shared IsaacLab teleop engine with the correct
task, single environment, CPU MuJoCo physics and GPU RTX visualizer. Extra options
and Hydra overrides are forwarded. `--background modeled|splats|0` selects the
authored greenhouse (default), optional partial Gaussian scan, or no background.

## Controls

- Hold **LB/L1** to enable gamepad commands. Left stick translates XY; right stick
  translates Z and controls yaw; D-pad controls roll/pitch.
- **Cross/A**, while enabled, toggles gripper open/closed. Options/Start resets.
- Release L1 to stop incremental motion, retaining the gripper state. Release
  and press it again after reset. `ISAACLAB_TOMATO_GAMEPAD=/dev/input/js1` selects
  another Linux joystick.
- Keyboard: **W/S, A/D, Q/E** translate; **Z/X, T/G, C/V** rotate; **K** toggles
  the gripper; **R** resets. Focus the render window.

Approach the red `tomato_00` from its exposed side, close, pull away from the plant,
then carry to the tray and release. The viewer reports attachment load and detachment.
Motion of the flexible branch alone does not imply the fruit has detached.

For lower assumed breaking loads (changes task mechanics):

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/tomato_pick/teleop_tomato.py \
  --mode gamepad 'env.scene.plant.separation_force_scale=0.5'
```

## Headless scripted pick

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/tomato_pick/teleop_tomato.py \
  --mode demo --output /tmp/tomato-pick.json
```

Choose a fresh output filename. This executes physical approach, grasp, pull,
carry, release and reset, with assertions for red-fruit delivery and no collateral
fruit detachment. No hidden grasp attachment or fruit teleport is used.

## Scope and assets

The plant has segmented rigid stems with passive bending joints, skinned stem
visuals, rigid leaves, and 16 independently breakable fruit attachments. Default
red/orange/green breaking loads are 1.5/2.5/4 N, with filtering and 30 ms dwell.
A 0.8 m/s breeze along +X is active at startup. Geometry, stiffness and thresholds
are approximate priors, not a calibrated botanical model. Only the native CPU
MuJoCo path is the supported default; the prior GPU path failed stability checks.

The plant is distributed as `plant/plant_visual.usdz`: visual geometry, materials,
textures, typed binding/calibration metadata, precomputed leaf wind areas, and
the original inline MJCF stored in a USD string attribute. Newton imports that
string directly; no XML, JSON, NPZ, or leaf-library sidecars are required.
Renderer resources resolve inside the USDZ archives without manual extraction.
The authored greenhouse is visual-only;
the optional Gaussian greenhouse is a limited-view reconstruction with holes in
unobserved viewpoints. That optional scan is CC-BY-NC-SA-3.0; derived leaf library
assets carry CC BY-SA 3.0 attribution. Notices are preserved in each asset's
`TaskData/Notices` USD metadata.
