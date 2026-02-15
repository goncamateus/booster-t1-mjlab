from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_task.reach_env import reach_env_cfg
from mjlab_task.rl_cfg import booster_t1_ppo_runner_cfg
from mjlab_task.stand_env import stand_env_cfg

register_mjlab_task(
    task_id="T1-Stand-v0",
    env_cfg=stand_env_cfg(),
    play_env_cfg=stand_env_cfg(play=True),
    rl_cfg=booster_t1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="T1-Reach-v0",
    env_cfg=reach_env_cfg(),
    play_env_cfg=reach_env_cfg(play=True),
    rl_cfg=booster_t1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
