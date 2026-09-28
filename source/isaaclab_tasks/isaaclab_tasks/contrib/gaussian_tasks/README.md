# Gaussian twin, tomato, and berry CLI tasks

This directory owns the shared installer, asset download/packaging/validation
tools, USD metadata helpers, and dependency patch for these contributed tasks.
Berry and tomato implementations and teleop CLIs remain in the sibling
`berry_pick/` and `tomato_pick/` directories. Gaussian-twin environment
configuration remains in the existing `stack/` task.

The former `scripts/tasks/` tools have moved here. Run the commands below from
the IsaacLab repository root; no task-specific implementation is kept in the
repository-wide `scripts/` directory.

This branch contains the task implementations and launchers. No `astra-rec`
checkout, Carsten POC checkout, or local source symlink is needed. **Live berry
rendering additionally requires the authorized OVRTX 0.6 / OVStage 0.3 wheels**
described below; they are not currently on public PyPI. Large task assets are distributed separately on
Nucleus and downloaded to a local cache. Ordinary Isaac Lab robot/table assets
still use Isaac Lab's configured NVIDIA asset server/cache.

## 1. Install the task runtime

Supported deployment: Linux x86-64, Python 3.12, NVIDIA RTX GPU with a compatible
CUDA/RTX driver, `git`, and `uv`. Interactive commands need a graphical display;
gamepad commands need a readable Linux `/dev/input/js*` device. First startup may
take several minutes for downloads, CUDA compilation, and renderer initialization.

For all three rendered tasks, use the internal renderer option from the root of
this IsaacLab branch. It requires access to NVIDIA Artifactory (NVIDIA network
or VPN, with any credentials required by your organization):

```bash
bash source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/setup.sh --renderer-internal
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-tasks"
```

The wheels are runtime dependencies, **not Nucleus task assets**. Setup installs
the exact published pair below, not an authoring-machine build. To check access
without changing an existing environment:

```bash
uv --no-config pip install --python .venv-tasks/bin/python --dry-run --no-deps \
  'ovrtx==0.6.0.dev382408+mr50035.92a010ff' ovstage==0.3.0.382327 \
  --index-url https://artifactory.nvidia.com/artifactory/api/pypi/ct-omniverse-pypi/simple
```

Use `artifactory.nvidia.com`, not `urm.nvidia.com`: the latter's indexes did not
expose these builds when checked on 2026-09-23. Keep the exact pins; an unbounded
version constraint can select a different development build or legacy version series.

**Validation status:** resolution of both pinned packages passed. Installation
and live-render verification of this published pair are still pending: the test
download was stopped after sustained transfer rates of approximately 20–30 KB/s.
The earlier local 0.6.0 / 0.3.0.0 builds passed live berry rendering; this does not
prove that the newly published pair behaves identically.

Packman also distributes the native OVRTX package (not a Python installation):

```bash
packman install ovrtx 0.6.0.dev382408+mr50035.92a010ff.manylinux_2_35_x86_64 \
  --remote cloudfront --link-path .tools/task-runtime/ovrtx-mr50035
```

That command assumes your Packman configuration defines the NVIDIA `cloudfront`
remote. The tested internal mirror configuration did not resolve this package;
the alternate CloudFront remote did. Packman is **not required** by the task
installer: the Python wheel carries its native runtime, and OVStage is installed
separately by setup. Do not replace the wheels with only the native archive.

For an existing authorized offline distribution, the earlier
`--renderer-wheels DIR` option remains available. It expects
`ovrtx-0.6.0-py3-none-manylinux_2_35_x86_64.whl` and
`ovstage-0.3.0.0-py3-none-manylinux_2_35_x86_64.whl` in that directory.

The script creates an isolated `.venv-tasks`; it does not modify `.venv-kitless`,
your sibling Newton checkout, or global Python. It installs the branch's frozen
base environment, then these task-specific dependencies:

| Dependency | Pinned task version |
| --- | --- |
| Newton | upstream `d37f4d3d341ccce1e06a1dff21e9a054759b4855` + branch-owned Gaussian viewer patch |
| Warp | 1.17.0 |
| MuJoCo / MuJoCo-Warp | 3.12.0 / 3.12.0 |
| Newton USD schemas | 0.5.0 |
| OVRTX / OVStage (internal install) | 0.6.0.dev382408+mr50035.92a010ff / 0.3.0.382327 |
| OVRTX / OVStage (public fallback) | 0.5.0.377615 / 0.2.0.377349; tomato, Gaussian twin, berry physics-only |
| USD provider | `usd-exchange==2.3.0` from the base lockfile; do not also install `usd-core` |

Newton's Gaussian viewer patch is included in `dependencies/newton-gaussian-viewer.patch`.
It reproduces the runtime changes between upstream `d37f4d3d` and the previously
used `28befda824c47ab92a6ca498a47361610d5de23b`, without requiring that unpublished
commit. Newton is Apache-2.0; the patch retains its source copyright/license headers.
Only runtime source changes are included, not Newton example assets or experiments.

The setup also installs this branch's `isaaclab_teleop` source package for shared
configuration types. The full Isaac Sim / XR / CloudXR stack is not required for
these keyboard/gamepad commands.

Without either renderer option, setup installs the public fallback. The berry CLI
then permits `--no_render` but explicitly rejects live rendering: in our test,
0.5 read back all dynamic arrays correctly while showing a ghosted berry.
Static rendering did not expose that failure. OVRTX 0.6 rendered the same live
scene correctly. This is separate from the repaired SH coefficients.

**Use `uv run --no-sync` for these tasks.** A normal `uv sync` restores the general
branch dependency pins, which do not include this task runtime. Rerun setup to
restore it. Setup validates the dedicated Newton checkout before applying its
patch and refuses a different revision. It never resets an existing checkout.

## 2. Download the assets (after the upload is published)

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/assets.py --task all
```

Default source:
`omniverse://content.ov.nvidia.com/Users/nicolasm@nvidia.com/isaaclab_assets/tasks/`

Default local destination: `~/.cache/isaaclab/tasks/`. Access to the Nucleus folder
is required. The tool reads only; it never publishes or modifies Nucleus. It
verifies every downloaded file against the bundle's SHA256 manifest, reuses
matching files, and refuses mismatched existing files unless `--replace` is given.

Download one task using `--task berry`, `--task tomato`, or `--task gaussian_twin`.
For a different cache location, set this variable **both when downloading and
when launching**:

```bash
export ISAACLAB_TASK_ASSET_ROOT=/path/to/local/tasks
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/assets.py --task all
```

To test a local copy of the upload before it is on Nucleus:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/assets.py --source /tmp/tasks \
  --destination /path/to/local/tasks --task all
```

The published bundle contains **only `.usd`, `.usda`, `.usdc`, or `.usdz` files**.
No NPZ, JSON, XML, shader, or texture sidecars need to be uploaded or downloaded
separately. The inventory itself is `task_assets_manifest.usda`.

Keep the local cache variables as filesystem paths, not `omniverse://` URLs.
Tasks read the USDZ packages directly through USD; shaders and textures resolve
inside the archives. No manual extraction or authoring source checkout is needed.

**Migration from the earlier bundle:** publish the new USD-only directory as a
replacement, not an overlay retaining old sidecars. Use a fresh local cache or
`--replace` for changed files. The downloader never deletes unrelated legacy
files; this version of the tasks does not read them.

## 3. Run a task

All commands below run from the IsaacLab root, with `UV_PROJECT_ENVIRONMENT` set
as above. One environment is supported for tomato and berry.

### Berry: deformable Gaussian fruit, continuous grasp

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/berry_pick/teleop_berry.py --mode keyboard
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/berry_pick/teleop_berry.py --mode gamepad
```

Use `--berry raspberry|blackberry|blueberry|strawberry` (default raspberry),
`--gamepad_device /dev/input/js0`, or `--width 960 --height 540` as needed.
All four assets are included; this CLI selects one berry per scene.
See the [berry README](../berry_pick/README.md)
for controls, scripted demos, and the known light-grasp slipping limitation.

### Tomato: articulated plant, detachable fruit

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/tomato_pick/teleop_tomato.py --mode keyboard
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/tomato_pick/teleop_tomato.py --mode gamepad
```

The authored greenhouse is the default. Use `--background 0` to omit it, or
`--background splats` for the optional partial Gaussian greenhouse scan.
See the [tomato README](../tomato_pick/README.md)
for controls, plant mechanics, and scripted picking.

### Gaussian twin: Franka manipulation of Gaussian toys

```bash
uv run --no-sync python scripts/environments/teleoperation/teleop_se3_agent.py \
  --task IsaacContrib-Stack-Gaussian-Twin-Franka-IK-Rel-Newton \
  --num_envs 1 --device cuda:0 --visualizer newton_rtx \
  --teleop_device gamepad --disable_external_cameras --no-auto_launch_cloudxr
```

Use `--teleop_device keyboard` for keyboard control. Assets default to
`$ISAACLAB_TASK_ASSET_ROOT/gaussian_twin/packages` (or the default cache root).
Without an asset override, the first sorted package is used. To choose a toy:

```bash
export ISAACLAB_GAUSSIAN_TWIN_ASSET="$HOME/.cache/isaaclab/tasks/gaussian_twin/packages/baked.bublik_octopus_package.usda"
```

Adjust this example path if using a custom cache. `ISAACLAB_GAUSSIAN_TWIN_DIR`
selects a different package directory; `ISAACLAB_GAUSSIAN_TWIN_NUM_OBJECTS` and
`ISAACLAB_GAUSSIAN_TWIN_NUM_SLOTS` retain the existing task behavior. The bundle
also includes six `vbd_4mm_512/*_skinned_vbd_tet.usda` packages; point `DIR` there
to use that alternative. Do not combine both package families in one directory.

The default is deformable VBD coupling. Existing `--rigid-simulation` and
`--dexsuite-simulation stable-kinematic|fast-kinematic` options remain available.
The optional `--simplicits-simulation` mode additionally requires the experimental
Kaolin Newton/RKPM integration; it is **not installed by this setup**. Its six
`*_simplicits_rkpm.usda` assets are included, but the default VBD/rigid commands
do not need Kaolin. Do not substitute the unrelated PyPI `kaolin` placeholder.

The Gaussian-twin task starts in the EBC lounge by default, using
`gaussian_twin/background/ebc/aligned.usda` from the shared task cache. This is
the original `point_cloud_aligned_new.usda` alignment, with its neighboring
`point_cloud.usd` scan (all 7,227,085 Gaussians retained). The older alignment
places the room away from the workcell; the newer alignment's two negative scale axes apply a
180-degree rotation, not a mirror. The robot remains at the task origin on its
table; a closer startup camera frames both the workcell and room.

The room is **visual only**: it does not introduce scanned furniture collisions
or change the task's robot/table physics. Its full scan adds substantial render
cost. Use `ISAACLAB_DISABLE_ALIGNED_BACKGROUND=1` for lightweight teleoperation
(`ISAACLAB_ENABLE_ALIGNED_BACKGROUND=0` is also supported). Set
`ISAACLAB_ALIGNED_BACKGROUND_USD=/path/to/another/aligned.usda` to override it.
Toy overrides such as `ISAACLAB_GAUSSIAN_TWIN_DIR` do not change the background
cache location.

The earlier Nova Carter/Galileo room is retained as
`gaussian_twin/background/aligned.usda`, referencing `nova_carter-galileo.usdz`.
No background references an authoring-machine path.

### Finite headless startup checks

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/smoke.py --task gaussian_twin --render
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/smoke.py --task tomato --render
```

These run five steps and exit; omit `--render` for physics-only startup. The berry
CLI provides `--mode idle --steps 30 --no_render` and `--verify_render` separately.

For a visual startup check, add `--capture /tmp/gaussian-twin-startup.png` to the
Gaussian-twin command. The path must not already exist. Optional
`--eye X Y Z --lookat X Y Z` camera coordinates are in meters; capture warms the
frozen scene for 32 render frames without stepping physics further.

To append EBC to an older USD-only bundle:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/add_ebc_background.py \
  --source /path/to/EBC_Newton_Demo --bundle /path/to/tasks
```

This preserves existing assets and updates the USD checksum manifest. The old
manifest is backed up beside the bundle as `tasks-manifest-before-ebc.usda`;
it is not part of the Nucleus upload. Existing destinations/backups are not overwritten.

## Upload layout and licensing

Copy the **contents** of the staged tasks directory (currently
`/mnt/data/isaac_lab_poc/tasks/`, originally `/tmp/tasks/`) into the Nucleus
`.../isaaclab_assets/tasks/` directory; do not add another nested `tasks` level:

```text
tasks/
  task_assets_manifest.usda
  berry/raspberry/raspberry.usdz
  berry/blackberry/blackberry.usdz
  berry/blueberry/blueberry.usdz
  berry/strawberry/strawberry.usdz
  tomato/plant/plant_visual.usdz
  tomato/greenhouse_modeled/{greenhouse.usdz,lighting.usda}
  tomato/greenhouse/greenhouse.usdz
  gaussian_twin/{packages,vbd_4mm_512,background}/...
  gaussian_twin/attribution.usdc
```

Attribution is stored as USD metadata; materials and textures travel inside
USDZ archives. Physics arrays and configuration are typed USD attributes, not
NPZ/JSON blobs. Tomato retains the original inline MJCF as a USD string attribute
for Newton's importer, preserving its calibrated mechanics without an XML file.
The optional Gaussian greenhouse is **CC-BY-NC-SA-3.0**; derived
tomato leaf assets include **CC BY-SA 3.0** material. Berry scan attribution is
preserved per asset. These assets are not relicensed by the IsaacLab code license;
respect their individual terms and access restrictions. Gaussian toy/background
data should remain within the owner's authorized distribution scope.

For future upload bundles, stage the USD-only directories and run:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/validate_assets.py /path/to/tasks
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/assets.py --manifest /path/to/tasks
```

The manifest command refuses non-USD files and an existing manifest. Publish a matching manifest
whenever asset contents change. No reconstruction/training environments, raw
datasets, videos, validation runs, or dependency wheels belong in the asset upload.

## Authoring / appearance repair

End users do **not** run asset preparation. To reproduce the migration from the
original mixed-file bundle and Carsten's authoring inputs, use a fresh directory:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/package_usd.py \
  --source /path/to/original/tasks --output /path/to/new/tasks \
  --poc /path/to/carsten_experiments
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/validate_assets.py /path/to/new/tasks
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/assets.py --manifest /path/to/new/tasks
```

The raspberry repair rebuilds exterior SH from `assets/raw/raspberry/scene.ply`,
reproduces the original weighted merging and initial rotation, then applies the
material rotation omitted when the settled pose was baked. Every prepared
Gaussian (including the interiors), covariance, opacity and tissue array is
retained. The SH basis is checked independently against SciPy's complex harmonics.
This fixes a verified preparation error; it does not certify an exact match to
Carsten's offline video. Camera, exposure, sampling and physics are unchanged by this repair.

## Validation

Validated on Linux / RTX 6000 Ada:

- Berry: all four USDZ loaders preserved the original arrays exactly, except the
  intentionally repaired raspberry SH. Physics/reset and array transport passed.
  Live raspberry rendering passed with the installed OVRTX 0.6 / OVStage 0.3
  runtime and the branch-owned Newton patch. The 0.5 display failure is guarded.
- Tomato: complete scripted pick/delivery/reset passed; offscreen greenhouse
  rendering passed. The task-local keyboard CLI reached its teleoperation loop
  and was stopped with a timed interrupt (not a manual picking validation).
- Gaussian twin: finite default VBD physics/render startup passed, including the
  optional aligned background. The checks do not establish long-horizon grasp
  quality for every solver mode or toy.
- Original mixed-file bundle: 433 files / 4,327,026,236 bytes were verified during
  the initial portability work. The USD-only v2 bundle initially replaced it with 29 assets
  plus one USD inventory. Adding EBC brings this to 31 assets and one inventory;
  91 USD/material/archive references were checked. Renderer-provided built-in
  MDL modules are excluded from local-file requirements. Relocated-cache berry
  startup/reset and Gaussian-twin rendering passed.
- USD-only asset-download, metadata, SH and renderer-version tests passed (14 tests, including
  the live quaternion/tint kernel). The omitted-rotation regression was also tested with
  rotation disabled: both rotation checks failed as expected.
- Gaussian configuration tests cover the current friction default of 2, explicit
  friction overrides, interactive solver profiles, and the default EBC camera/background.

The Nucleus upload itself remains the publisher's step. Local-copy download and
checksum verification were tested; access to the not-yet-published remote bundle
cannot be certified in advance. The berry contact/slipping limitation remains.
The pinned internal renderer pair is now published and resolves from the direct
Artifactory index. Its installation/live-render check remains pending because
of the slow download noted above; the successful 0.6 run used the earlier
existing installed build.
