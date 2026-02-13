from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)

def booster_t1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster T1 task."""
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.01,
            learning_rate=3e-4, # Match train.py
            num_learning_epochs=5,
            num_mini_batches=4, # 4 * ? = batch size
        ),
        experiment_name="booster_t1_velocity",
        max_iterations=1000, # Start small
    )
