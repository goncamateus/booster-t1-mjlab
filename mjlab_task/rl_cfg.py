from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


def booster_t1_ppo_runner_cfg(exp_name: str, num_iterations: int) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster T1 task."""
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            actor_hidden_dims=(512, 256, 256, 128),
            critic_hidden_dims=(512, 256, 256, 128),
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.01,
            learning_rate=3e-4,  # Match train.py
            num_learning_epochs=10,
            num_mini_batches=4,  # 4 * ? = batch size
        ),
        experiment_name=exp_name,
        max_iterations=num_iterations,
    )
