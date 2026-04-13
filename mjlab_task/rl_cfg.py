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
            noise_std_type="scalar",
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=(512, 256, 256, 128),
            critic_hidden_dims=(512, 256, 256, 128),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.01,
            learning_rate=3e-4,
            num_learning_epochs=10,
            num_mini_batches=4,
        ),
        experiment_name=exp_name,
        max_iterations=num_iterations,
    )
