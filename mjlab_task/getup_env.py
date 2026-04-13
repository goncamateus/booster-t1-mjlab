"""Getup (fall recovery) environment for Booster T1 on RoboCup field.

1 robot + 1 field per env. Adapted from mjlab_playground getup task.
Scene uses assets/booster_t1/field.xml instead of flat plane.
"""

import os

from mujoco import MjSpec

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers import ObservationGroupCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import getup_mdp
from .getup_mdp import SettleRelativeJointPositionActionCfg
from .robot_cfg import BoosterT1Cfg

_TORSO_HEIGHT = 0.67
_WAIST_HEIGHT = 0.55

_POSTURE_STD = {
    r".*_Hip_Roll": 0.08,
    r".*_Hip_Yaw": 0.08,
    r".*_Hip_Pitch": 0.12,
    r".*_Knee_Pitch": 0.15,
    r".*_Ankle_Pitch": 0.2,
    r".*_Ankle_Roll": 0.2,
    r"(AAHead_yaw|Head_pitch)": 0.15,
    r"(Waist|.*_Shoulder.*|.*_Elbow.*)": 0.5,
}

_FIELD_XML = os.path.join(os.path.dirname(__file__), "../assets/booster_t1/field.xml")


def _merge_field_spec(spec: MjSpec) -> None:
    field_spec = MjSpec.from_file(_FIELD_XML)
    site = spec.worldbody.add_site(name="field_site", pos=(0, 0, 0))
    spec.attach(field_spec, site=site)
    spec.delete(site)


def getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    actor_terms = {
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
            noise=Unoise(n_min=-0.03, n_max=0.03),
            params={"biased": True},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    critic_terms = {
        **actor_terms,
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions = {
        "joint_pos": SettleRelativeJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.6,
            settle_steps=50,
        )
    }

    events = {
        "reset_fallen_or_standing": EventTermCfg(
            func=getup_mdp.reset_fallen_or_standing,
            mode="reset",
            params={
                "fall_probability": 1.0 if play else 0.6,
                "fall_height": 0.8,
                "velocity_range": 0.5,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=mdp.randomize_encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
                "operation": "add",
                "field": "body_ipos",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
            },
        ),
        "geom_friction_slide": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
                "operation": "abs",
                "axes": [0],
                "ranges": (0.3, 1.5),
                "shared_random": True,
            },
        ),
    }

    foot_geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
    )
    events["foot_friction_spin"] = EventTermCfg(
        mode="startup",
        func=mdp.randomize_field,
        domain_randomization=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
            "operation": "abs",
            "distribution": "log_uniform",
            "axes": [1],
            "ranges": (1e-4, 2e-2),
            "shared_random": True,
        },
    )
    events["foot_friction_roll"] = EventTermCfg(
        mode="startup",
        func=mdp.randomize_field,
        domain_randomization=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
            "operation": "abs",
            "distribution": "log_uniform",
            "axes": [2],
            "ranges": (1e-5, 5e-3),
            "shared_random": True,
        },
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    rewards = {
        "orientation": RewardTermCfg(func=getup_mdp.orientation_reward, weight=1.0),
        "torso_height": RewardTermCfg(
            func=getup_mdp.height_reward,
            weight=1.0,
            params={
                "desired_height": _TORSO_HEIGHT,
                "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
            },
        ),
        "waist_height": RewardTermCfg(
            func=getup_mdp.height_reward,
            weight=1.0,
            params={
                "desired_height": _WAIST_HEIGHT,
                "asset_cfg": SceneEntityCfg("robot", body_names=("Waist",)),
            },
        ),
        "posture": RewardTermCfg(
            func=getup_mdp.gated_posture_reward,
            weight=1.0,
            params={
                "orientation_threshold": 0.01,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "std": _POSTURE_STD,
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=0.0),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "self_collisions": RewardTermCfg(
            func=getup_mdp.self_collision_cost,
            weight=-0.1,
            params={"sensor_name": self_collision_cfg.name},
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "energy": TerminationTermCfg(
            func=getup_mdp.energy_termination,
            params={"threshold": float("inf"), "settle_steps": 50},
        ),
    }

    curriculum = {
        "action_rate_weight": CurriculumTermCfg(
            func=getup_mdp.reward_curriculum,
            params={
                "reward_name": "action_rate_l2",
                "stages": [
                    {"step": 0, "weight": -0.01},
                    {"step": 600 * 24, "weight": -0.05},
                    {"step": 900 * 24, "weight": -0.08},
                    {"step": 1200 * 24, "weight": -0.1},
                ],
            },
        ),
        "joint_vel_weight": CurriculumTermCfg(
            func=getup_mdp.reward_curriculum,
            params={
                "reward_name": "joint_vel_l2",
                "stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 900 * 24, "weight": -0.005},
                    {"step": 1200 * 24, "weight": -0.008},
                    {"step": 1500 * 24, "weight": -0.01},
                ],
            },
        ),
        "energy_threshold": CurriculumTermCfg(
            func=getup_mdp.termination_curriculum,
            params={
                "termination_name": "energy",
                "stages": [
                    {"step": 900 * 24, "params": {"threshold": 3000.0}},
                    {"step": 1200 * 24, "params": {"threshold": 2000.0}},
                    {"step": 1500 * 24, "params": {"threshold": 1500.0}},
                    {"step": 1700 * 24, "params": {"threshold": 1000.0}},
                    {"step": 2200 * 24, "params": {"threshold": 700.0}},
                ],
            },
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={"robot": BoosterT1Cfg()},
            sensors=(self_collision_cfg,),
            num_envs=1,
            extent=2.0,
            spec_fn=_merge_field_spec,
            terrain=None,
        ),
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="Trunk",
            distance=1.5,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            njmax=200,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=6.0,
    )

    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.episode_length_s = 20.0

    return cfg
