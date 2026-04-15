from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


def booster_t1_ppo_runner_cfg(
    exp_name: str, num_iterations: int
) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster T1 task."""
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            noise_std_type="log",
            actor_obs_normalization=False,
            critic_obs_normalization=True,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=exp_name,
        max_iterations=num_iterations,
    )
