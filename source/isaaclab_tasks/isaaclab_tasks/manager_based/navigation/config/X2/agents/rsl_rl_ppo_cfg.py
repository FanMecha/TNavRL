# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg


@configclass
class NavigationEnvPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # batch_size = (num_envs * num_steps_per_env) / num_mini_batches
    # batch_size = 8192 always can balance the velocity and accuracy of training for CNN
    # For example:
    # num_envs = 4096, num_steps_per_env = 16, num_mini_batches = 8
    # num_envs = 2048, num_steps_per_env = 16, num_mini_batches = 4
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "X2_navigation"
    empirical_normalization = True
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",

        rnn_type="lstm_sru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
)