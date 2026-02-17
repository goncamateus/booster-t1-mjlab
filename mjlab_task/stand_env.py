import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import ObservationGroupCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from .robot_cfg import BoosterT1Cfg


def torso_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_com_pos_w[:, asset_cfg.body_ids, 2].sum()


def feet_on_ground(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.found is not None
    return (sensor_data.found > 0).float().sum(dim=1)


class T1StandCfgGen:
    obs = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
    }

    reward_set = {
        # Set params and weights
        "pose",
        # Set params["asset_cfg"].body_names
        "upright",
        "body_ang_vel",
        # Unchange
        "soft_landing",
        "dof_pos_limits",
        "action_rate_l2",
        # Set weights
        "track_linear_velocity",
        "track_angular_velocity",
        # New
        "feet_on_ground",
        "torso_height",
    }

    def __init__(self):
        self.cfg: ManagerBasedRlEnvCfg = make_velocity_env_cfg()

    def setup_scene(self):
        """Configure the scene for the Booster T1 environment."""
        # Set robot
        self.cfg.scene.entities = {"robot": BoosterT1Cfg()}
        self.cfg.scene.terrain.terrain_type = "plane"
        self.cfg.scene.terrain.terrain_generator = None

        # Add contact sensor for feet (required by standard velocity task observations/rewards)
        self.cfg.scene.sensors = (
            ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(
                    mode="body",
                    pattern=("left_foot_link", "right_foot_link"),
                    entity="robot",
                ),
                fields=("force", "torque", "found"),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                track_air_time=True,
                history_length=5,
                reduce="netforce",
            ),
        )

    def setup_viewer(self):
        """Configure the viewer for the Booster T1 environment."""
        self.cfg.viewer.body_name = "base"
        self.cfg.viewer.body_name = "Trunk"

    def _setup_pose_reward(self):
        """Configure the pose reward for the Booster T1 environment."""
        self.cfg.rewards["pose"].params["std_standing"] = {".*": 10}
        self.cfg.rewards["pose"].params["std_walking"] = {".*": 0}
        self.cfg.rewards["pose"].params["std_running"] = {".*": 0}
        self.cfg.rewards["pose"].weight = 10

    def _setup_body_rewards(self):
        """Configure body-related rewards for the Booster T1 environment."""
        self.cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
        self.cfg.rewards["upright"].weight = 10

        self.cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)
        self.cfg.rewards["body_ang_vel"].weight = 10

    def _setup_velocity_tracking_rewards(self):
        """Configure velocity tracking rewards for the Booster T1 environment."""
        self.cfg.rewards["track_linear_velocity"].weight = -1
        self.cfg.rewards["track_angular_velocity"].weight = -1

    def _setup_torso_reward(self):
        """Configure the feet on ground reward for the Booster T1 environment."""
        self.cfg.rewards["torso_height"] = RewardTermCfg(
            func=torso_height,
            weight=10.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=(".*",), body_names=("Trunk",)
                )
            },
        )

    def _setup_feet_on_ground_reward(self):
        """Configure the feet on ground reward for the Booster T1 environment."""
        self.cfg.rewards["feet_on_ground"] = RewardTermCfg(
            func=feet_on_ground,
            weight=1.0,
            params={"sensor_name": "feet_ground_contact"},
        )

    def setup_rewards(self):
        """Configure rewards for the Booster T1 environment."""
        self._setup_pose_reward()
        self._setup_body_rewards()
        self._setup_velocity_tracking_rewards()
        self._setup_torso_reward()
        self._setup_feet_on_ground_reward()
        self.cfg.rewards = {key: self.cfg.rewards[key] for key in self.reward_set}

    def choose_observation(self):
        self.obs = {
            key: self.obs[key]
            for key in (
                "base_lin_vel",
                "base_ang_vel",
                "projected_gravity",
                "joint_pos",
                "joint_vel",
            )
        }

    def setup_observations(self):
        """Configure observations for the Booster T1 environment."""
        self.choose_observation()
        self.cfg.observations = {
            "policy": ObservationGroupCfg(
                terms=self.obs,
                concatenate_terms=True,
                enable_corruption=True,
                nan_policy="error",
            ),
            "critic": ObservationGroupCfg(
                terms=self.obs,
                concatenate_terms=True,
                enable_corruption=True,
                nan_policy="error",
            ),
        }

    def generate(self, play: bool = False) -> ManagerBasedRlEnvCfg:
        """Generate the Booster T1 environment configuration."""
        self.setup_scene()
        self.setup_viewer()
        self.setup_observations()
        self.setup_rewards()
        joint_pos_action = self.cfg.actions["joint_pos"]
        joint_pos_action.scale = 1.0
        self.cfg.curriculum: dict[str, CurriculumTermCfg] = {}
        if play:
            self.cfg.episode_length_s = 20.0  # longer for play
        return self.cfg


def stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Generate the Booster T1 Stand environment configuration."""
    return T1StandCfgGen().generate(play=play)
