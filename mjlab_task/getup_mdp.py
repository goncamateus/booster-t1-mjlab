"""MDP functions for the Getup environment, adapted from mjlab_playground getup task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_UP_VEC = torch.tensor([0.0, 0.0, -1.0])


@dataclass(kw_only=True)
class RelativeJointPositionActionCfg(BaseActionCfg):
    """Control joints with deltas based on current state."""

    settle_steps: int = 0

    def __post_init__(self) -> None:
        self.transmission_type = TransmissionType.JOINT

    def build(self, env: ManagerBasedRlEnv) -> "RelativeJointPositionAction":
        return RelativeJointPositionAction(self, env)


class RelativeJointPositionAction(BaseAction):
    def apply_actions(self) -> None:
        current = self._entity.data.joint_pos[:, self._target_ids]
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = current + self._processed_actions - encoder_bias
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


def _is_upright(
    asset: Entity,
    orientation_threshold: float,
) -> torch.Tensor:
    gravity = asset.data.projected_gravity_b
    up = _UP_VEC.to(gravity.device)
    error = torch.sum(torch.square(up - gravity), dim=-1)
    return (error < orientation_threshold).float()


class SettleRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    settle_steps: int = 0

    def build(self, env: ManagerBasedRlEnv) -> "SettleRelativeJointPositionAction":
        return SettleRelativeJointPositionAction(self, env)


class SettleRelativeJointPositionAction(RelativeJointPositionAction):
    def __init__(
        self,
        cfg: SettleRelativeJointPositionActionCfg,
        env: ManagerBasedRlEnv,
    ):
        super().__init__(cfg=cfg, env=env)
        self._settle_steps = cfg.settle_steps

    def apply_actions(self) -> None:
        current_pos = self._entity.data.joint_pos[:, self._target_ids]
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = current_pos + self._raw_actions * self._scale - encoder_bias
        if self._settle_steps > 0:
            in_window = self._env.episode_length_buf < self._settle_steps
            was_fallen = self._env.extras.get("settle_mask", in_window)
            settling = (in_window & was_fallen).unsqueeze(-1)
            target = torch.where(settling, current_pos - encoder_bias, target)
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


def reset_fallen_or_standing(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    fall_probability: float = 0.6,
    fall_height: float = 0.5,
    velocity_range: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    n = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]

    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert soft_joint_pos_limits is not None

    fall_mask = torch.rand(n, device=env.device) < fall_probability
    if "settle_mask" not in env.extras:
        env.extras["settle_mask"] = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    env.extras["settle_mask"][env_ids] = fall_mask

    root_states = default_root_state[env_ids].clone()

    random_quat = torch.randn(n, 4, device=env.device)
    random_quat = F.normalize(random_quat, dim=-1)

    fallen_positions = env.scene.env_origins[env_ids].clone()
    fallen_positions[:, 2] += fall_height

    fallen_velocities = sample_uniform(
        -velocity_range, velocity_range, (n, 6), env.device
    )

    standing_positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    standing_positions[:, 2] += 0.02

    mask = fall_mask.unsqueeze(-1)
    positions = torch.where(mask, fallen_positions, standing_positions)
    orientations = torch.where(mask, random_quat, root_states[:, 3:7])
    velocities = torch.where(mask, fallen_velocities, root_states[:, 7:13])

    asset.write_root_link_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)

    joint_limits = soft_joint_pos_limits[env_ids]
    random_joint_pos = sample_uniform(
        joint_limits[..., 0],
        joint_limits[..., 1],
        joint_limits[..., 0].shape,
        env.device,
    )

    joint_pos = torch.where(mask, random_joint_pos, default_joint_pos[env_ids].clone())
    joint_vel = torch.where(
        mask,
        sample_uniform(-velocity_range, velocity_range, joint_pos.shape, env.device),
        default_joint_vel[env_ids].clone(),
    )

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def orientation_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    up = _UP_VEC.to(gravity.device)
    error = torch.sum(torch.square(up - gravity), dim=-1)
    return torch.exp(-2.0 * error)


def height_reward(
    env: ManagerBasedRlEnv,
    desired_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    import math

    asset: Entity = env.scene[asset_cfg.name]
    height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(-1)
    clamped = torch.clamp(height, max=desired_height)
    return (torch.exp(clamped) - 1.0) / (math.exp(desired_height) - 1.0)


class gated_posture_reward:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        _, joint_names = asset.find_joints(
            cfg.params["asset_cfg"].joint_names,
        )

        _, _, std = resolve_matching_names_values(
            data=cfg.params["std"],
            list_of_strings=joint_names,
        )
        self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: dict[str, float],
        orientation_threshold: float = 0.01,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del std
        asset: Entity = env.scene[asset_cfg.name]
        gate = _is_upright(asset, orientation_threshold)
        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)
        return gate * torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


def energy_termination(
    env: ManagerBasedRlEnv,
    threshold: float = float("inf"),
    settle_steps: int = 0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    power = torch.sum(
        torch.abs(
            asset.data.actuator_force[:, asset_cfg.actuator_ids]
            * asset.data.joint_vel[:, asset_cfg.joint_ids]
        ),
        dim=-1,
    )
    past_settle = env.episode_length_buf > settle_steps
    return past_settle & (power > threshold)


def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str = "self_collision",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    contact_forces = sensor.data.force
    collision_force_magnitude = contact_forces.abs().sum(dim=-1)
    return -collision_force_magnitude.mean(dim=-1)


def reward_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    stages: list[dict],
) -> torch.Tensor:
    del env_ids
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in stages:
        if env.common_step_counter > stage["step"]:
            reward_term_cfg.weight = stage["weight"]
    return torch.tensor([reward_term_cfg.weight])


def termination_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    termination_name: str,
    stages: list[dict],
) -> torch.Tensor:
    del env_ids
    termination_term_cfg = env.termination_manager.get_term_cfg(termination_name)
    for stage in stages:
        if env.common_step_counter > stage["step"]:
            for key, value in stage["params"].items():
                setattr(termination_term_cfg, key, value)
    return torch.tensor([0.0])
