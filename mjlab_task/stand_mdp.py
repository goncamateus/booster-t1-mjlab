"""MDP functions for the Stand environment, adapted from mjlab_playground getup task."""

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
# Target gravity in body frame when upright (gravity points -z in world).
_UP_VEC = torch.tensor([0.0, 0.0, -1.0])


# ---------------------------------------------------------------------------
# SettleJointPositionAction — suppresses actions during early steps after fall
# ---------------------------------------------------------------------------

class SettleJointPositionActionCfg(JointPositionActionCfg):
    settle_steps: int = 50  # controller timesteps to suppress (1s at 50Hz for T1)

    def __post_init__(self):
        super().__post_init__()

    def build(self, env: "ManagerBasedRlEnv") -> "SettleJointPositionAction":
        return SettleJointPositionAction(self, env)


class SettleJointPositionAction(JointPositionAction):
    """JointPositionAction that holds the default pose for settle_steps after a fallen reset.

    The reset event stores fallen env ids on the env object; the action picks them up
    during ``reset()``, which is called right after events in the env's ``_reset_idx``.
    """

    def __init__(self, cfg: SettleJointPositionActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.settle_steps = cfg.settle_steps
        self._settle_counter = torch.zeros(env.num_envs, dtype=torch.int, device=env.device)
        # Store reference for the reset event to write to
        self._fallen_env_ids: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # Pick up fallen env ids that reset_fallen_or_standing stored
        if self._fallen_env_ids is not None and len(self._fallen_env_ids) > 0:
            self._settle_counter[self._fallen_env_ids] = self.settle_steps
        self._fallen_env_ids = None

    def apply_actions(self) -> None:
        # Handle settling: hold default pose for fallen envs
        settling = self._settle_counter > 0
        if settling.any():
            settling_f = settling.float().unsqueeze(1)
            self._processed_actions = (
                self._processed_actions * (1.0 - settling_f)
                + self._offset * settling_f
            )
        # Decrement counter
        if settling.any():
            self._settle_counter[settling] -= 1

        super().apply_actions()


# ---------------------------------------------------------------------------
# Reset event
# ---------------------------------------------------------------------------

def reset_fallen_or_standing(
    env: "ManagerBasedRlEnv",
    *,
    fall_probability: float = 0.6,
    fall_height: float = 0.8,
    velocity_range: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset robot as either fallen or standing.

    Fallen envs: dropped from fall_height with random orientation & random joints.
    Standing envs: default pose lifted slightly (~2 cm).

    The ``fallen_env_ids`` tensor is stored on the ``SettleJointPositionAction``
    instance so the action can hold the default pose during settling.
    """
    asset: Entity = env.scene[asset_cfg.name]
    device = env.device

    num_envs = env.num_envs
    is_fallen = torch.rand(num_envs, device=device) < fall_probability
    fallen_ids = torch.arange(num_envs, device=device)[is_fallen]
    standing_ids = torch.arange(num_envs, device=device)[~is_fallen]

    # ---- fallen ----
    if fallen_ids.numel() > 0:
        fallen_env_ids = fallen_ids.to(torch.int)
        n_fallen = len(fallen_env_ids)

        # Random quaternion orientation
        quat = _random_unit_quat(n_fallen, device)

        # Root position at fall height
        origins = env.scene.env_origins[fallen_ids]
        root_pos = origins.clone()
        root_pos[:, 2] += fall_height

        # Random joint positions across full range
        joint_limits = asset.data.joint_pos_limits[fallen_ids]  # (n_fallen, n_joints, 2)
        lo = joint_limits[:, :, 0]
        hi = joint_limits[:, :, 1]
        joint_pos = lo + (hi - lo) * torch.rand_like(lo)

        # Zero velocities
        vel = torch.zeros(n_fallen, asset.num_dof - 6, device=device)

        # Write root state
        asset.write_root_link_pose_to_sim(
            torch.cat([root_pos, quat], dim=-1), env_ids=fallen_env_ids
        )
        asset.write_root_link_velocity_to_sim(
            torch.zeros(n_fallen, 6, device=device), env_ids=fallen_env_ids
        )

        # Write joint state
        asset.write_joint_state_to_sim(
            joint_pos, vel, env_ids=fallen_env_ids
        )

    # ---- standing ----
    if standing_ids.numel() > 0:
        standing_env_ids = standing_ids.to(torch.int)

        # Default root state lifted 2 cm
        default_root = asset.data.default_root_state[standing_ids].clone()
        origins = env.scene.env_origins[standing_ids]
        default_root[:, 0:3] += origins
        default_root[:, 2] += 0.02  # 2 cm z bump

        # Zero velocities
        vel = torch.zeros(len(standing_ids), asset.num_dof - 6, device=device)
        default_jpos = asset.data.default_joint_pos[standing_ids]

        asset.write_root_link_pose_to_sim(
            default_root[:, 0:7], env_ids=standing_env_ids
        )
        asset.write_root_link_velocity_to_sim(
            torch.zeros(len(standing_ids), 6, device=device), env_ids=standing_env_ids
        )
        asset.write_joint_state_to_sim(
            default_jpos[:, :asset.num_joints], vel, env_ids=standing_env_ids
        )

    # Signal fallen envs to the settle action
    action = env.action_manager._terms.get("joint_pos")
    if isinstance(action, SettleJointPositionAction):
        action._fallen_env_ids = fallen_ids.to(torch.int) if fallen_ids.numel() > 0 else None


def _random_unit_quat(n: int, device: torch.device) -> torch.Tensor:
    """Generate random unit quaternions."""
    u1 = torch.rand(n, device=device)
    u2 = torch.rand(n, device=device) * 2 * math.pi
    u3 = torch.rand(n, device=device) * 2 * math.pi
    a = (1 - u1).sqrt() * (2 * u2).sin()
    b = (1 - u1).sqrt() * (2 * u2).cos()
    c = u1.sqrt() * (2 * u3).sin()
    d = u1.sqrt() * (2 * u3).cos()
    return torch.stack([a, b, c, d], dim=1)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def orientation_reward(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for upright orientation."""
    asset: Entity = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    up = _UP_VEC.to(gravity.device)
    error = torch.sum(torch.square(up - gravity), dim=-1)
    return torch.exp(-2.0 * error)


class gated_posture_reward:
    """Reward for returning to default pose, gated on being upright."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, std = resolve_matching_names_values(
            data=cfg.params["std"],
            list_of_strings=joint_names,
        )
        self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        std: dict[str, float],
        orientation_threshold: float = 0.01,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del std  # resolved in __init__
        asset: Entity = env.scene[asset_cfg.name]
        gravity = asset.data.projected_gravity_b
        up = _UP_VEC.to(gravity.device)
        error = torch.sum(torch.square(up - gravity), dim=-1)
        gate = (error < orientation_threshold).float()

        current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_sq = torch.square(current_pos - desired_pos)
        return gate * torch.exp(-torch.mean(error_sq / (self.std**2), dim=1))
