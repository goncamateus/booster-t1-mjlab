from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from mjlab_task.reach_mdp import reach_goal_reward, rel_goal_pos
from mjlab_task.stand_env import T1StandCfgGen


class T1ReachCfgGen(T1StandCfgGen):
    obs_goal = ObservationTermCfg(func=rel_goal_pos, params={"command_name": "goal"})

    def __init__(self):
        super().__init__()

    def setup_scene(self):
        """Configure the scene for the Booster T1 environment."""
        super().setup_scene()

    def choose_observation(self):
        self.obs = {
            key: self.obs[key]
            for key in (
                "base_lin_vel",
                "base_ang_vel",
                "projected_gravity",
                "joint_pos",
                "joint_vel",
                "command",
            )
        }

    def setup_rewards(self):
        # Start with stand rewards
        self.cfg.rewards["pose"].params["std_standing"] = {".*": 1}
        self.cfg.rewards["pose"].params["std_walking"] = {".*": 2}
        self.cfg.rewards["pose"].params["std_running"] = {".*": 5}
        self.cfg.rewards["reach_goal"] = RewardTermCfg(
            func=reach_goal_reward,
            weight=10.0,
            params={"command_name": "goal", "std": 0.5},
        )


def reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return T1ReachCfgGen().generate(play=play)

    # Ensure other rewards use the new command name
    for reward in cfg.rewards.values():
        if "command_name" in reward.params and reward.params["command_name"] == "twist":
            reward.params["command_name"] = "goal"

    # Disable curricula that depend on velocity commands
    cfg.curriculum = {}

    return cfg
