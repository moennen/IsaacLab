#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Dedicated, reproducible Kitless task environment; never updates .venv-kitless.
set -euo pipefail
TASK_RENDERER_WHEELS=""
TASK_RENDERER_INTERNAL=false
if [[ "${1:-}" == "--help" && $# == 1 ]]; then
    echo "Usage: bash source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/setup.sh [--renderer-internal | --renderer-wheels /authorized/wheel/directory]"
    echo "Live berry rendering requires OVRTX 0.6 / OVStage 0.3 Linux wheels."
    echo "--renderer-internal installs pinned builds from NVIDIA Artifactory (network access required)."
    exit 0
fi
if [[ "${1:-}" == "--renderer-wheels" && $# == 2 ]]; then
    TASK_RENDERER_WHEELS="$(cd -- "$2" && pwd)"
    TASK_OVRTX_WHEEL="$TASK_RENDERER_WHEELS/ovrtx-0.6.0-py3-none-manylinux_2_35_x86_64.whl"
    TASK_OVSTAGE_WHEEL="$TASK_RENDERER_WHEELS/ovstage-0.3.0.0-py3-none-manylinux_2_35_x86_64.whl"
    [[ -f "$TASK_OVRTX_WHEEL" && -f "$TASK_OVSTAGE_WHEEL" ]] || {
        echo "Missing validated OVRTX 0.6 / OVStage 0.3 Linux wheels in $TASK_RENDERER_WHEELS" >&2
        exit 1
    }
elif [[ "${1:-}" == "--renderer-internal" && $# == 1 ]]; then
    TASK_RENDERER_INTERNAL=true
elif [[ $# != 0 ]]; then
    echo "Usage: bash source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/setup.sh [--renderer-internal | --renderer-wheels /authorized/wheel/directory]" >&2
    exit 1
fi
TASK_TOOLS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_LAB_ROOT="$(cd -- "$TASK_TOOLS_ROOT/../../../../.." && pwd)"
TASK_NEWTON_ROOT="$TASK_LAB_ROOT/.tools/task-runtime/newton"
TASK_NEWTON_COMMIT=d37f4d3d341ccce1e06a1dff21e9a054759b4855
TASK_PATCH="$TASK_TOOLS_ROOT/dependencies/newton-gaussian-viewer.patch"
export UV_PROJECT_ENVIRONMENT="$TASK_LAB_ROOT/.venv-tasks"
cd "$TASK_LAB_ROOT"

uv sync --frozen --python 3.12
if [[ ! -e "$TASK_NEWTON_ROOT" ]]; then
    mkdir -p "$TASK_NEWTON_ROOT"
    git -C "$TASK_NEWTON_ROOT" init -q
    git -C "$TASK_NEWTON_ROOT" fetch --depth 1 https://github.com/newton-physics/newton.git "$TASK_NEWTON_COMMIT"
    git -C "$TASK_NEWTON_ROOT" checkout --detach FETCH_HEAD
fi
if [[ "$(git -C "$TASK_NEWTON_ROOT" rev-parse HEAD)" != "$TASK_NEWTON_COMMIT" ]]; then
    echo "Unexpected Newton checkout in $TASK_NEWTON_ROOT; refusing to overwrite it." >&2
    exit 1
fi
if git -C "$TASK_NEWTON_ROOT" apply --reverse --check "$TASK_PATCH" 2>/dev/null; then
    echo "Newton Gaussian viewer patch is already applied."
else
    git -C "$TASK_NEWTON_ROOT" diff --quiet
    git -C "$TASK_NEWTON_ROOT" apply --check "$TASK_PATCH"
    git -C "$TASK_NEWTON_ROOT" apply "$TASK_PATCH"
fi
# Override only the dedicated environment after installing the branch lockfile.
# --no-deps avoids introducing a second pxr provider beside usd-exchange.
uv --no-config pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps \
    -e "$TASK_NEWTON_ROOT" \
    -e "$TASK_LAB_ROOT/source/isaaclab_teleop" \
    warp-lang==1.17.0 newton-usd-schemas==0.5.0 \
    mujoco==3.12.0 mujoco-warp==3.12.0
if [[ "$TASK_RENDERER_INTERNAL" == true ]]; then
    uv --no-config pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps \
        'ovrtx==0.6.0.dev382408+mr50035.92a010ff' ovstage==0.3.0.382327 \
        --index-url https://artifactory.nvidia.com/artifactory/api/pypi/ct-omniverse-pypi/simple
elif [[ -n "$TASK_RENDERER_WHEELS" ]]; then
    uv --no-config pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps \
        "$TASK_OVRTX_WHEEL" "$TASK_OVSTAGE_WHEEL"
else
    uv --no-config pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps \
        ovrtx==0.5.0.377615 ovstage==0.2.0.377349
    echo "Public renderer installed for tomato/Gaussian twin and berry physics-only."
    echo "Berry rendering requires setup.sh --renderer-internal or --renderer-wheels DIR." >&2
fi
echo "Ready. Export UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT and use uv run --no-sync."
