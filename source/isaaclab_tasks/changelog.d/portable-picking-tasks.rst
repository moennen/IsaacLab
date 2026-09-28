Added
^^^^^

* Added self-contained berry and tomato task CLIs, a shared local asset cache,
  verified Nucleus downloads, and reproducible Kitless setup instructions. Replaced
  machine-specific task asset defaults with the shared cache while retaining
  explicit per-task environment overrides. Added a default cached package directory
  for Gaussian twins. Existing deployments can retain their asset-root overrides.

* Moved shared installation, asset CLI tools, USD metadata/cache helpers, and the
  Newton dependency patch into ``isaaclab_tasks.contrib.gaussian_tasks`` beside
  the berry and tomato task implementations. Updated commands and imports from
  the former ``scripts/tasks`` and ``isaaclab_tasks.utils`` locations; use the new
  task-local README and ``gaussian_tasks/setup.sh`` entry point.
