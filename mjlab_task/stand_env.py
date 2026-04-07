"""Stand environment for Booster T1, adapted from mjlab_playground getup task."""
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers import ObservationGroupCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import stand_mdp
from .robot_cfg import BoosterT1Cfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# ---------------------------------------------------------------------------
# Helper reward / termination functions (kept inline, not in separate file)
# ---------------------------------------------------------------------------

_UP_VEC = torch.tensor([0.0, 0.0, -1.0])


def _projected_gravity(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    return asset.data.projected_gravity_b


def torso_height(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_com_pos_w[:, asset_cfg.body_ids, 2].mean(dim=1)


def waist_height_reward(
    env: "ManagerBasedRlEnv",
    desired_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    height = asset.data.body_com_pos_w[:, asset_cfg.body_ids, 2].mean(dim=1)
    clamped = torch.clamp(height, max=desired_height)
    return (torch.exp(clamped) - 1.0) / (math.exp(desired_height) - 1.0)


def feet_on_ground(
    env: "ManagerBasedRlEnv", sensor_name: str = "feet_ground_contact"
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    return (sensor.data.found > 0).float().sum(dim=1)


# ---------------------------------------------------------------------------
# T1StandCfgGen
# ---------------------------------------------------------------------------

class T1StandCfgGen:

    _POSTURE_STD = {
        ".*_Hip_Roll": 0.08,
        ".*_Hip_Yaw": 0.08,
        ".*_Hip_Pitch": 0.12,
        ".*_Knee_Pitch": 0.15,
        ".*_Ankle_Pitch": 0.2,
        ".*_Ankle_Roll": 0.2,
        "AAHead_yaw.*": 0.15,
        "Head_pitch.*": 0.15,
        "Waist.*": 0.5,
        ".*_Shoulder.*": 0.5,
        ".*_Elbow.*": 0.5,
    }

    # --- actor (policy) observations ---
    _actor_obs = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=_projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
            params={"biased": True},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    # critic gets extra state: joint_pos (no noise / bias), base_lin_vel
    _critic_obs = {
        **_actor_obs,
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
    }

    reward_set = {
        "pose",
        "upright",
        "body_ang_vel",
        "soft_landing",
        "dof_pos_limits",
        "action_rate_l2",
        "feet_on_ground",
        "torso_height",
        "posture",
        "waist_height",
    }

    def __init__(self):
        self.cfg = make_velocity_env_cfg()

    # ---- scene ----
    def setup_scene(self):
        self.cfg.scene.entities = {"robot": BoosterT1Cfg()}
        self.cfg.scene.terrain.terrain_type = "plane"
        self.cfg.scene.terrain.terrain_generator = None
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

    # ---- viewer ----
    def setup_viewer(self):
        self.cfg.viewer.body_name = "Trunk"

    # ---- actions ----
    def setup_actions(self):
        """Replace JointPositionAction with SettleJointPositionAction for fall recovery.

        We patch the config dict after creation so that the action's `build()` method
        returns our custom action that suppresses actions during settle_steps.
        """
        action = stand_mdp.SettleJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=1.0,
            use_default_offset=True,
        )
        # Add settle_steps to the config (the custom action reads it directly)
        action.settle_steps = 50  # 1.0s at 50Hz for T1
        self.cfg.actions["joint_pos"] = action

    # ---- events (reset) ----
    def setup_events(self, play: bool = False):
        # Replace base-velocity events with fall-recovery style reset.
        # play mode: always start fallen (100 % fall rate) — easier to test recovery.
        fall_prob = 1.0 if play else 0.6
        self.cfg.events = {
            "reset_fallen_or_standing": EventTermCfg(
                func=stand_mdp.reset_fallen_or_standing,
                mode="reset",
                params={
                    "fall_probability": fall_prob,
                    "fall_height": 0.8,
                    "velocity_range": 0.5,
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                },
            ),
            "encoder_bias": self.cfg.events.pop("encoder_bias", None),
            "base_com": self.cfg.events.pop("base_com", None),
        }
        # Remove any leftover keys that are None
        self.cfg.events = {k: v for k, v in self.cfg.events.items() if v is not None}

    # ---- rewards ----
    def _setup_pose_reward(self):
        self.cfg.rewards["pose"].params["std_standing"] = {".*": 10}
        self.cfg.rewards["pose"].params["std_walking"] = {".*": 0}
        self.cfg.rewards["pose"].params["std_running"] = {".*": 0}
        self.cfg.rewards["pose"].weight = 100

    def _setup_body_rewards(self):
        self.cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
        self.cfg.rewards["upright"].weight = 10
        self.cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)
        self.cfg.rewards["body_ang_vel"].weight = -1

    def _setup_torso_reward(self):
        self.cfg.rewards["torso_height"] = RewardTermCfg(
            func=torso_height,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=(".*",), body_names=("Trunk",)
                )
            },
        )

    def _setup_waist_height_reward(self):
        self.cfg.rewards["waist_height"] = RewardTermCfg(
            func=waist_height_reward,
            weight=1.0,
            params={
                "desired_height": 0.55,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=(".*",), body_names=("Waist",)
                ),
            },
        )

    def _setup_feet_on_ground_reward(self):
        self.cfg.rewards["feet_on_ground"] = RewardTermCfg(
            func=feet_on_ground,
            weight=10.0,
            params={"sensor_name": "feet_ground_contact"},
        )

    def _setup_posture_reward(self):
        self.cfg.rewards["posture"] = RewardTermCfg(
            func=stand_mdp.gated_posture_reward,
            weight=1.0,
            params={
                "orientation_threshold": 0.01,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "std": self._POSTURE_STD,
            },
        )

    def setup_rewards(self):
        for fn in (
            self._setup_pose_reward,
            self._setup_body_rewards,
            self._setup_torso_reward,
            self._setup_waist_height_reward,
            self._setup_feet_on_ground_reward,
            self._setup_posture_reward,
        ):
            fn()
        # Keep only desired rewards
        self.cfg.rewards = {k: self.cfg.rewards[k] for k in self.reward_set}

    # ---- observations ----
    def setup_observations(self, play: bool = False):
        actor_terms = dict(self._actor_obs)
        critic_terms = dict(self._critic_obs)
        if play:
            # Disable observation noise when evaluating
            for term in actor_terms.values():
                term.noise = None
            for term in critic_terms.values():
                term.noise = None
        self.cfg.observations = {
            "policy": ObservationGroupCfg(
                terms=actor_terms,
                concatenate_terms=True,
                enable_corruption=True,
                nan_policy="error",
            ),
            "critic": ObservationGroupCfg(
                terms=critic_terms,
                concatenate_terms=True,
                enable_corruption=True,
                nan_policy="error",
            ),
        }

    # ---- terminations ----
    def setup_terminations(self):
        max_tilt_rad = 0.12
        self.cfg.terminations = {
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        }

    # ---- metrics (not supported in current mjlab) ----
    def setup_metrics(self):
        pass  # MetricsTermCfg not available; success tracked via rewards

    # ---- assemble ----
    def generate(self, play: bool = False):
        """Generate the Booster T1 Stand environment configuration."""
        self.setup_scene()
        self.setup_viewer()
        self.setup_actions()
        self.setup_events(play=play)
        self.setup_observations(play=play)
        self.setup_terminations()
        self.setup_rewards()
        self.setup_metrics()
        self.cfg.curriculum = {}
        self.cfg.commands = {}
        if play:
            self.cfg.episode_length_s = 20.0
        else:
            self.cfg.episode_length_s = 6.0
        return self.cfg


def stand_env_cfg(play: bool = False):
    """Generate the Booster T1 Stand environment configuration."""
    return T1StandCfgGen().generate(play=play)
