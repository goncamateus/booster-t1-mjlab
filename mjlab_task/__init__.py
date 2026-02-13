from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import booster_t1_env_cfg, booster_t1_reach_env_cfg
from .rl_cfg import booster_t1_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Booster-v0",
    env_cfg=booster_t1_env_cfg(),
    play_env_cfg=booster_t1_env_cfg(play=True),
    rl_cfg=booster_t1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Booster-Reach-v0",
    env_cfg=booster_t1_reach_env_cfg(),
    play_env_cfg=booster_t1_reach_env_cfg(play=True),
    rl_cfg=booster_t1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
