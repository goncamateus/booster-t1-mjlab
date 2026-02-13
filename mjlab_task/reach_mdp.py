from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class GoalCommand(CommandTerm):
    """Command term for a target (x, y, z) location."""

    cfg: GoalCommandCfg

    def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["distance_to_goal"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def _resample_command(self, env_ids: torch.Tensor):
        r = torch.empty(len(env_ids), device=self.device)
        self.goal_pos_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.x)
        self.goal_pos_w[env_ids, 1] = r.uniform_(*self.cfg.ranges.y)
        self.goal_pos_w[env_ids, 2] = self.cfg.ranges.z

    def _update_command(self):
        pass

    def _update_metrics(self):
        robot = self._env.scene[self.cfg.entity_name]
        self.metrics["distance_to_goal"] = torch.norm(
            self.goal_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=1
        )

    @property
    def command(self) -> torch.Tensor:
        return self.goal_pos_w


@dataclass(kw_only=True)
class GoalCommandCfg(CommandTermCfg):
    """Configuration for the goal command."""

    entity_name: str = "robot"

    @dataclass
    class Ranges:
        x: tuple[float, float] = (-5.0, 5.0)
        y: tuple[float, float] = (-5.0, 5.0)
        z: float = 0.0

    ranges: Ranges = field(default_factory=Ranges)

    def build(self, env: ManagerBasedRlEnv) -> GoalCommand:
        return GoalCommand(self, env)


def rel_goal_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Observation: goal position in robot's local frame."""
    command = env.command_manager.get_command(command_name)
    robot = env.scene["robot"]
    # Goal in world frame relative to robot's roots position
    rel_pos_w = command - robot.data.root_link_pos_w
    # Rotate into robot's local frame
    return quat_apply_inverse(robot.data.root_link_quat_w, rel_pos_w)


def reach_goal_reward(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward for being close to the goal (x, y)."""
    command = env.command_manager.get_command(command_name)
    robot = env.scene["robot"]
    # We only care about x, y for now as per user request
    dist_sq = torch.sum(
        torch.square(command[:, :2] - robot.data.root_link_pos_w[:, :2]), dim=1
    )
    return torch.exp(-dist_sq / std**2)
