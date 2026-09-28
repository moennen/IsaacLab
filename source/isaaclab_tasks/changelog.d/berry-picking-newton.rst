Added
^^^^^

* Added a task-local Franka berry-picking integration with continuous gamepad grasp,
  GPU MPM tissue, and full interior/exterior Gaussian rendering. The implementation
  and teleoperation entry point were included in ``isaaclab_tasks.contrib.berry_pick``;
  assets were distributed separately in the shared USD-only task bundle.
* Added independently selectable DLAA and RTPT sample-count overrides for visual
  comparison without changing the default rendering or physical state.
