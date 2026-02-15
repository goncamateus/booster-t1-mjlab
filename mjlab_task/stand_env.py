from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import ObservationGroupCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from .robot_cfg import BoosterT1Cfg


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

    reward_set = {"pose", "upright"}

    def __init__(self):
        self.cfg: ManagerBasedRlEnvCfg = make_velocity_env_cfg()

    def setup_scene(self):
        """Configure the scene for the Booster T1 environment."""
        # Set robot
        self.cfg.scene.entities = {"robot": BoosterT1Cfg()}

        # Add contact sensor for feet (required by standard velocity task observations/rewards)
        self.cfg.scene.sensors = (
            ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(
                    mode="body",
                    pattern=("left_foot_link", "right_foot_link"),
                    entity="robot",
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                track_air_time=True,
            ),
        )

    def setup_viewer(self):
        """Configure the viewer for the Booster T1 environment."""
        self.cfg.viewer.body_name = "base"
        self.cfg.viewer.body_name = "Trunk"

    def setup_rewards(self):
        """Configure rewards for the Booster T1 environment."""
        self.cfg.rewards["pose"].params["std_standing"] = {".*": 0.1}
        self.cfg.rewards["pose"].params["std_walking"] = {".*": 0}
        self.cfg.rewards["pose"].params["std_running"] = {".*": 0}
        self.cfg.rewards["pose"].weight = 1
        self.cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
        self.cfg.rewards["upright"].weight = 10
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
