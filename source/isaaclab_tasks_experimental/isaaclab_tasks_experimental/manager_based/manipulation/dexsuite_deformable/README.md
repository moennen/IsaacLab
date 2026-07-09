# DexSuite deformable manipulation

This task simulates a heterogeneous set of Gaussian-splat assets with Newton
VBD. Each selected asset must have the same number of simulation vertices;
tetrahedron counts may differ.

## Heterogeneous-asset constraints

### Common simulation vertex count

The common vertex count is an Isaac Lab batching constraint, not a VBD
constraint. Newton stores all particles in one flat model and can simulate
different vertex counts. The task, however, exposes deformable state and
targets as dense tensors with shape `(num_envs, particles_per_body, ...)` for
RL observations, resets, kinematic targets, and Gaussian skinning. Its
builder records the particle count from environment 0 and rejects any later
environment with a different count. Supporting variable counts would require
padded tensors plus a per-environment valid-node mask (or ragged offset-based
views) throughout the data API, reset kernels, observations, rewards, and
visualizer. Tetrahedron counts are not exposed through those dense node
tensors, so they may vary today.

### Native VBD versus the MuJoCo/Warp--VBD manager

Newton's native `SolverVBD` is a unified coupled solver: it can advance VBD
particles and AVBD rigid bodies together. The reason this task does not use it
for the complete Kuka/Allegro scene is robot-actuator support, not soft-body
coupling. The robot's Isaac Lab configuration uses armature, joint friction,
effort limits, and implicit drive control. `SolverVBD` supports target
stiffness/damping and feedforward force, but not joint armature, joint
friction, joint effort limits, or MuJoCo's joint target modes. Using it would
therefore change or omit the configured robot dynamics and actuator limits.

`NewtonCoupledMJWarpVBDManager` retains those MuJoCo/Warp articulation and
actuator features while using VBD for the soft body. Select it with
`presets=stable_two_way` (or `presets=fast_two_way`). It collides first,
injects normal and Coulomb-friction reactions into the rigid-body force buffer,
steps MuJoCo/Warp with those reactions, then steps VBD using the same contacts.
The default `stable_kinematic` preset instead uses Featherstone--VBD with a
kinematically driven robot. It is the default because it follows robot position
targets exactly and is more stable for early policy training; the two-way
MuJoCo/Warp choice makes the arm dynamically respond to deformable contact and
consequently needs actuator/contact tuning.

## Asset preparation

This section fully reproduces the six packaged toy assets used by this
workspace. Run it from `IsaacLab/`. It requires:

- the six source USDZ files at `/mnt/gogn/data/isaac_lab_poc/ExportedToys`;
- the Isaac Lab environment (`env_isaaclab`); and
- VoMP's inference-only environment and its `weights/inference.json`
  checkpoint configuration. Install the latter once from the VoMP checkout:

```bash
cd /mnt/dev/VoMP
bash install_micromamba_inference.sh
huggingface-cli download nvidia/PhysicalAI-Simulation-VoMP-Model \
  --local-dir /mnt/dev/VoMP/weights
```

The source toys are Z-up. Keep that convention in both the PLY export and VBD
build; using `--source-z-up` in only one of the two commands will misalign the
visual splats and material field.

```bash
cd /nv/dev/isaaclab-lebesgue/IsaacLab

TOY_SOURCE_DIR=/mnt/gogn/data/isaac_lab_poc/ExportedToys
ASSET_ROOT="$PWD/outputs/dexsuite_toys_collision_4mm_512"
TARGET_NUM_VERTICES=512
TARGET_MAX_EXTENT=0.335
COLLISION_SHELL=0.004
SKINNING_CHUNK_SIZE=128

GAUSSIAN_USDS=(
  "$TOY_SOURCE_DIR/baked.BluehairRagdoll.usdz"
  "$TOY_SOURCE_DIR/baked.bublik_octopus.usdz"
  "$TOY_SOURCE_DIR/baked.knit_meow.usdz"
  "$TOY_SOURCE_DIR/baked.mer_elephant.usdz"
  "$TOY_SOURCE_DIR/baked.stink_raccoon.usdz"
  "$TOY_SOURCE_DIR/baked.sunflower_baby.usdz"
)
```

### 1. Export the Gaussian fields for VoMP

The exporter writes standard 3D Gaussian-splat PLY fields (`x`, `y`, `z`,
opacity, scales, rotations, and spherical harmonics). It is not a mesh export.
The parameters above reproduce the current training-compatible set: 512 VBD
particles per asset, a 0.335 m maximum extent, and a 4 mm collision shell.

```bash
./isaaclab.sh -p -m isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.tools.export_gaussian_usd_to_ply \
  "${GAUSSIAN_USDS[@]}" \
  --output-dir "$ASSET_ROOT/vomp_ply" \
  --target-max-extent "$TARGET_MAX_EXTENT" \
  --source-z-up
```

### 2. Infer one VoMP material file per PLY

Run VoMP on each exported PLY. The output files contain a structured
`voxel_data` array with `x`, `y`, `z`, Young's modulus, Poisson ratio, and
density. Keep the material list in exactly the same order as `GAUSSIAN_USDS`.

```bash
VOMP_ROOT=/mnt/dev/VoMP
PLY_DIR="$ASSET_ROOT/vomp_ply"
MATERIAL_DIR="$PLY_DIR/materials"
mkdir -p "$MATERIAL_DIR"

PLY_FILES=()
for USD in "${GAUSSIAN_USDS[@]}"; do
  PLY_FILES+=("$PLY_DIR/$(basename "${USD%.*}").ply")
done

VOMP_MATERIALS=()
for PLY in "${PLY_FILES[@]}"; do
  NAME="$(basename "${PLY%.ply}")"
  MATERIAL="$MATERIAL_DIR/$NAME.npz"
  (
    cd "$VOMP_ROOT"
    micromamba run -n vomp-inference python scripts/extract_gaussian_materials.py \
      "$PLY" --output "$MATERIAL" --config weights/inference.json
  )
  VOMP_MATERIALS+=("$MATERIAL")
done
```

VoMP may normalize its sparse voxel field to an approximately unit longest
extent. Do not manually scale, translate, or rotate its NPZ output. The VBD
builder uniformly aligns the VoMP bounding box to the generated tet rest frame
before looking up material values.

### 3. Build the VBD tet proxies and skinned Gaussian packages

This uses the current training-compatible budget: 512 VBD particles per asset,
a 0.335 m maximum rest extent, and a 4 mm collision shell. The task's
`DEFORMABLE_PARTICLE_RADIUS` must use the same 4 mm value.

```bash
./isaaclab.sh -p -m isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_deformable.tools.build_gaussian_vbd_asset_set \
  "${GAUSSIAN_USDS[@]}" \
  --output-dir "$ASSET_ROOT" \
  --target-num-vertices "$TARGET_NUM_VERTICES" \
  --target-max-extent "$TARGET_MAX_EXTENT" \
  --collision-shell "$COLLISION_SHELL" \
  --source-z-up \
  --skinning-chunk-size "$SKINNING_CHUNK_SIZE" \
  --vomp-materials "${VOMP_MATERIALS[@]}" \
  --youngs-modulus-correction-factor 1e5
```

The command writes six entries to `$ASSET_ROOT/manifest.json`, one
legacy VBD source in `vbd_tets/`, and one Gaussian-skinned package in
`packaged/` for each input. With VoMP data, the builder assigns Young's
modulus and Poisson ratio at each tet centroid, converts them to Lamé
parameters (`newton:tetMu`, `newton:tetLambda`), samples density at vertices,
and writes mass-lumped vertex masses (`newton:particleMass`). Omit
`--vomp-materials` only when uniform fallback material is intended.

By default, `--youngs-modulus-correction-factor 1e5` rescales all supplied
VoMP Young's-modulus values by one shared factor so their combined arithmetic
mean is `1e5` Pa. This corrects VoMP's stiffness bias while preserving relative
material variation. Pass `--youngs-modulus-correction-factor 1.0` to use raw
VoMP values without correction.

### 4. Use the generated set in the task

```bash
export DEXSUITE_DEFORMABLE_ASSETS="$ASSET_ROOT/vbd_tets/baked.BluehairRagdoll_vbd_tet.usda,$ASSET_ROOT/vbd_tets/baked.bublik_octopus_vbd_tet.usda,$ASSET_ROOT/vbd_tets/baked.knit_meow_vbd_tet.usda,$ASSET_ROOT/vbd_tets/baked.mer_elephant_vbd_tet.usda,$ASSET_ROOT/vbd_tets/baked.stink_raccoon_vbd_tet.usda,$ASSET_ROOT/vbd_tets/baked.sunflower_baby_vbd_tet.usda"
export DEXSUITE_SKINNED_GAUSSIAN_ASSETS="$ASSET_ROOT/packaged/baked.BluehairRagdoll_skinned_vbd_tet.usda,$ASSET_ROOT/packaged/baked.bublik_octopus_skinned_vbd_tet.usda,$ASSET_ROOT/packaged/baked.knit_meow_skinned_vbd_tet.usda,$ASSET_ROOT/packaged/baked.mer_elephant_skinned_vbd_tet.usda,$ASSET_ROOT/packaged/baked.stink_raccoon_skinned_vbd_tet.usda,$ASSET_ROOT/packaged/baked.sunflower_baby_skinned_vbd_tet.usda"
export DEXSUITE_DEFORMABLE_ASSETS_ARE_Z_UP=1
export DEXSUITE_DEFORMABLE_PARTICLE_RADIUS=0.004
```

## Training and evaluation

Train the RSL-RL policy:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Dexsuite-Deformable-Kuka-Allegro-Lift-v0 presets=stable_two_way
```

Play a checkpoint (replace the checkpoint arguments with the desired run):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Dexsuite-Deformable-Kuka-Allegro-Lift-Play-v0 \
  --checkpoint /path/to/model.pt \
  presets=stable_two_way
```

Use the same physics preset that was used for training.  In particular, the
``2026-07-05_11-20-48/model_12500.pt`` checkpoint was trained with
``presets=fast_two_way``.

### Reference play command

The following is the canonical Kit-visualizer replay of the packaged toy set
with the `2026-07-05_11-20-48/model_12500.pt` checkpoint. It sets the six VBD
and skinned-Gaussian assets, forces constant gravity (disabling the gravity
curriculum and the variable-gravity event so the play episode uses a fixed
`-9.81` m/s²), and caps the episode at 4 s:

```bash
ASSET_ROOT="$PWD/outputs/dexsuite_toys_collision_4mm_512"

export DEXSUITE_DEFORMABLE_ASSETS="$ASSET_ROOT/vbd_tets/baked.BluehairRagdoll_vbd_tet.usda,$ASSET_ROOT/vbd_tets/baked.bublik_octopus_vbd_tet.usda,$ASSET_ROOT/vbd_tets/baked.knit_meow_vbd_tet.usda,$ASSET_ROOT/vbd_tets/baked.mer_elephant_vbd_tet.usda,$ASSET_ROOT/vbd_tets/baked.stink_raccoon_vbd_tet.usda,$ASSET_ROOT/vbd_tets/baked.sunflower_baby_vbd_tet.usda"
export DEXSUITE_SKINNED_GAUSSIAN_ASSETS="$ASSET_ROOT/packaged/baked.BluehairRagdoll_skinned_vbd_tet.usda,$ASSET_ROOT/packaged/baked.bublik_octopus_skinned_vbd_tet.usda,$ASSET_ROOT/packaged/baked.knit_meow_skinned_vbd_tet.usda,$ASSET_ROOT/packaged/baked.mer_elephant_skinned_vbd_tet.usda,$ASSET_ROOT/packaged/baked.stink_raccoon_skinned_vbd_tet.usda,$ASSET_ROOT/packaged/baked.sunflower_baby_skinned_vbd_tet.usda"
export DEXSUITE_DEFORMABLE_ASSETS_ARE_Z_UP=1
export DEXSUITE_DEFORMABLE_PARTICLE_RADIUS=0.004

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Dexsuite-Deformable-Kuka-Allegro-Lift-Kit-Play-v0 \
  --num_envs 1 \
  --visualizer kit \
  --checkpoint "$PWD/logs/rsl_rl/dexsuite_deformable_kuka_allegro_lift/2026-07-05_11-20-48/model_12500.pt" \
  presets=fast_two_way \
  env.sim.gravity=[0.0,0.0,-9.81] \
  env.events.variable_gravity=null \
  env.curriculum.gravity_adr=null \
  env.episode_length_s=4.0
```

Run the environment with a zero-action agent:

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Dexsuite-Deformable-Kuka-Allegro-Lift-Kit-Play-v0 --num_envs 1 --visualizer kit
```

## Rendering


### Standard deformable mesh path

The simulation proxy is a standard `UsdGeom.TetMesh` with a derived
`UsdGeom.Mesh` surface. Its visual mesh has one position per Newton particle,
so the shared Newton deformable integration can bind the mesh `points` array
to Fabric and update it directly from `particle_q` on the GPU. This is the
normal mesh-deformation path and does not require task-specific rendering
logic.

### Gaussian-splat path

A `ParticleField3DGaussianSplat` is not a mesh and has no standard USD skinning
or deformable-binding schema. It also has many more visual splats than VBD
vertices, and its splat centers generally lie on the visual surface while the
tet proxy is a coarse, inset volume. Therefore a direct particle-to-`points`
copy would be incorrect.

The asset builder stores four tet-vertex indices and barycentric weights for
each Gaussian center in the `newton:deformableSkin:*` attributes. The
task-specific visualizer evaluates that interpolation every replay frame and
writes the resulting centers to a `ParticleField3DGaussianSplat` prim. This is
why Gaussian replay has a dedicated mode: it is implementing a currently
missing renderer-visible Gaussian skinning bridge, rather than duplicating mesh
rendering code.

### Performance and limitations

For the Kit visualizer, the preferred path binds the authored Gaussian
`positions` arrays through Fabric and runs the barycentric skinning Warp kernel
entirely on the GPU. It avoids a device-to-host copy and is the fastest update
path available with the current USD particle-field renderer. If Fabric binding
is unavailable or incomplete, the visualizer falls back to copying centers to
the CPU and calling USD `positions.Set()` once per visible environment; this is
substantially slower. Keep `max_visible_envs` and
`max_gaussians_per_env` small for interactive replay.

The Newton visualizer is a debugging view, not a Gaussian renderer: it displays
the skinned centers as colored spheres/points. Both visualizers currently skin
centers only. Splat scale, orientation, and spherical-harmonic appearance stay
at their rest values, so large local stretch, shear, or rotation is not yet
reflected in the Gaussian covariance. A faster or more physically complete
path would require renderer support for GPU-resident Gaussian skinning,
including deformation of each splat's covariance, rather than only its center.
