Aligning a USD Background
=========================

``scripts/tools/align_scene.py`` opens a registered Isaac Lab task together with a USD
background asset, such as a Gaussian splat.  The background can be translated, rotated,
and scaled in the Kit window.  ``Save USDA`` writes a small composition layer containing
only a reference to the original asset and its transform; the source USD is not copied or
modified.

Run it from the Isaac Lab repository, for example:

.. code-block:: bash

   uv run isaaclab -p scripts/tools/align_scene.py \
       --task Isaac-Reach-Franka \
       --background-usd /absolute/path/to/background.usd \
       --output-usd /absolute/path/to/aligned_background.usda \
       --visualizer kit \
       --num-envs 1 \
       physics=newton_mjwarp presets=diffik

The task is loaded using the same configuration and backend overrides as the other
environment scripts.  The alignment controls affect the Kit stage immediately.  Save the
USDA layer, then reference that layer from the task's scene configuration instead of the
original background asset:

.. code-block:: python

   background = sim_utils.UsdFileCfg(
       usd_path="/absolute/path/to/aligned_background.usda",
       semantic_tags=[("class", "background")],
   )

The default wrapper prim is ``/World/GaussianBackground``.  Use ``--background-prim`` to
choose another absolute path.  The saved USDA uses a relative reference when possible, so
it remains relocatable with the source asset and alignment layer.

.. note::

   Alignment is a Kit/rendering workflow.  If the selected backend imports the background
   into its physics model, apply the saved layer before the simulation is initialized so
   the backend sees the final transform.
