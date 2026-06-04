# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for the experimental Kuka/Allegro deformable lift task."""

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import PresetCfg


POLICY_CFG = RslRlMLPModelCfg(
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8),
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    activation="elu",
)

CRITIC_CFG = RslRlMLPModelCfg(
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    activation="elu",
)

ALGO_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.0025,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=5.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class DexsuiteDeformableKukaAllegroPPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Base PPO runner tuned for state-only deformable manipulation."""

    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 250
    experiment_name = MISSING
    obs_groups = MISSING
    actor = MISSING
    critic = MISSING
    algorithm = MISSING


@configclass
class DexsuiteDeformableKukaAllegroPPORunnerCfg(PresetCfg):
    """Preset wrapper for future camera or privileged-observation variants."""

    default = DexsuiteDeformableKukaAllegroPPOBaseRunnerCfg().replace(
        experiment_name="dexsuite_deformable_kuka_allegro_lift",
        obs_groups={
            "actor": ["policy", "proprio", "deformable"],
            "critic": ["policy", "proprio", "deformable"],
        },
        actor=POLICY_CFG,
        critic=CRITIC_CFG,
        algorithm=ALGO_CFG,
    )
