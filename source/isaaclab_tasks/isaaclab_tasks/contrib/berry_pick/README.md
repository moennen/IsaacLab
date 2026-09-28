# Berry picking

Start with [runtime installation and Nucleus download](../gaussian_tasks/README.md).
The implementation and CLI are entirely in this task directory; no external
task checkout or Carsten source checkout is required. Assets default to the shared
local task cache. `ISAACLAB_BERRY_ASSET_ROOT` optionally overrides the berry folder.

Live rendering requires OVRTX 0.6 / OVStage 0.3; install the pinned internal builds with
`bash source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/setup.sh --renderer-internal` (NVIDIA Artifactory access required).
The public 0.5 renderer is rejected for this task because live Gaussian updates
can become invisible despite correct array readback. Physics-only `--no_render`
remains supported with the public runtime. See the shared guide for exact pins,
the Packman native package, and the offline wheel-directory alternative.

From the IsaacLab root, after setup:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-tasks"
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/berry_pick/teleop_berry.py --mode gamepad
```

Use `--mode keyboard` if no joystick is connected. Raspberry is default;
`--berry` also accepts blackberry, blueberry and strawberry. `--help` lists all
CLI options. Robot state uses the native CPU MuJoCo backend; MPM and Gaussians
run on CUDA device 0. An RTX GPU is required for rendering.

## Controls

- Hold **LB/L1** to enable gamepad motion and grasp commands.
- Left stick: XY translation. Right stick: Z translation and yaw. D-pad: roll/pitch.
- **RT/R2 closes**, **LT/L2 opens** continuously. Trigger pressure controls speed,
  up to 12 mm/s total aperture change. Release triggers to hold the commanded opening.
- Menu/Start resets robot and tissue. Release and press LB again after reset.
- Keyboard: **W/S, A/D, Q/E** translate; **Z/X, T/G, C/V** rotate; **K** closes,
  **J** opens, **R** resets. Focus the render window.

This is **position control, not force control**: holding a compressed opening can
continue deforming the berry. Opening the fingers relieves compression. There is
no automatic stop-at-contact behavior. The viewer has a follow-berry camera toggle.

## Scripted checks

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/berry_pick/teleop_berry.py \
  --mode idle --no_render --steps 30 --output /tmp/berry-smoke
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/berry_pick/teleop_berry.py \
  --mode pick --steps 540 --verify_reset --output /tmp/berry-pick
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/berry_pick/teleop_berry.py \
  --mode squash --steps 480 --output /tmp/berry-squash
```

Choose fresh output directories. Add `--window` for interactive scripted viewing,
or `--video` to record (requires ffmpeg). Scripts without `--window` render offscreen.
`--verify_render` checks native Gaussian publication; `--render_aa dlaa --rtpt_spp 4`
are optional sampling overrides, not guaranteed appearance fixes.

## Physical limitations

The 2 mm MPM contact grid currently causes grid-alignment-sensitive slipping in
light grasps. The scripted 13 mm pick can narrow tissue width by about 29% while
reporting negligible damage; it is not a validated gentle pick. Increasing finger
friction alone did not resolve this. This packaging refactor does not change the
contact model or conceal that limitation. Contact covers the inner fingertip pads,
not the entire arm. Materials and geometry scale are demo priors, not calibrated
fruit measurements. Other berry sizes require operator adjustment.

All prepared exterior/interior Gaussians and degree-three SH data are retained.
The runtime reads Gaussian appearance, tissue arrays, and simulation settings
directly from `<berry>/<berry>.usdz`. No NPZ or JSON files are needed. The USDZ
contains a USD layer with `/Berry/Gaussians`, `/Berry/Tissue`, typed `/Berry/TaskData`
metadata, and the MDL shader. Renderer resources resolve inside the USDZ archive.
Importing the USD in a generic Gaussian-capable viewer displays the
appearance; the task's MPM adapter instantiates the tissue simulation.

The raspberry SH coefficients were rebuilt from the original scan and rotated
through the settled material frame (about 59 degrees), a step missing from the
original `raspberry-squash` preparation. Geometry, opacity, interiors and physics
were preserved. `sh_rotation.py` and `rebuild_appearance.py` contain the offline
repair; the runtime performs no coefficient fitting or additional per-frame work.
See the shared CLI guide for authoring commands and the USD-only upload layout.
