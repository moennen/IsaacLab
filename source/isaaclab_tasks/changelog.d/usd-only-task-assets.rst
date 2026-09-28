Changed
^^^^^^^

* Changed berry and tomato tasks to load self-contained USDZ assets and published
  the shared download inventory as USD. Preserved Gaussian-twin USD packages and
  embedded attribution in USD metadata. Rebuilt raspberry SH from the source scan
  and applied the previously omitted settled material rotation without changing
  geometry or mechanics.

  **Breaking asset-layout change:** republish the USD-only v2 bundle and download
  to a fresh cache (or use ``source/isaaclab_tasks/isaaclab_tasks/contrib/gaussian_tasks/assets.py --replace``).
  Legacy NPZ/JSON/XML sidecars are no longer read. Shaders and textures are included inside USDZ.

* Required the validated OVRTX 0.6 / OVStage 0.3 runtime for live berry rendering
  after reproducing ghosted dynamic fields with public OVRTX 0.5. Added a
  ``--renderer-wheels`` installer option and a startup guard with migration guidance.
  Added ``--renderer-internal`` to install published OVRTX
  ``0.6.0.dev382408+mr50035.92a010ff`` and OVStage ``0.3.0.382327`` from NVIDIA
  Artifactory. Accepted this exact OVRTX prerelease in the startup guard.
  Renderer dependencies remained separate from the USD-only asset upload.
