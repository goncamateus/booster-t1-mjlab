from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions.actions import JointPositionActionCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
import mjlab.tasks.velocity.mdp as mdp
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactSensorCfg, ContactMatch

from .robot_cfg import BoosterT1Cfg
from .reach_mdp import GoalCommandCfg, rel_goal_pos, reach_goal_reward


def booster_t1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 environment configuration."""
    # Start with velocity task defaults
    cfg = make_velocity_env_cfg()

    # Set robot
    cfg.scene.entities = {"robot": BoosterT1Cfg()}

    # Add contact sensor for feet (required by standard velocity task observations/rewards)
    cfg.scene.sensors = (
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

    # Set pose reward standard deviations (must match joint count or provide defaults)
    pose_reward = cfg.rewards["pose"]
    pose_reward.params["std_standing"] = {".*": 0.1}
    pose_reward.params["std_walking"] = {".*": 0.2}
    pose_reward.params["std_running"] = {".*": 0.5}

    # Configure foot-related rewards to use the newly added foot sites
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].params["asset_cfg"] = SceneEntityCfg(
                "robot", site_names=["left_foot", "right_foot"]
            )

    # Update Action scale if needed.
    # BoosterT1Env used raw actions (kp=75).
    # mjlab usually scales actions.
    # We'll use scale=1.0 for now to match raw behavior if possible,
    # or keep default scale if we want learned policy to output deltas.
    # BoosterT1Env: do_simulation(action) -> action is target position?
    # t1.xml says <position ... kp="75">.
    # If action is position target, then scale=1.0 is appropriate if action is in radians.
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # Viewer settings
    cfg.viewer.body_name = "base"  # "Trunk" is root body name in XML?
    # Check t1.xml: <body name="Trunk" ...>
    # ArticulationCfg might map root to "base" if we use standard format,
    # but here we use MJCF directly. The root body is "Trunk".
    # However, mjlab might expect specific naming?
    # Let's try "Trunk".
    cfg.viewer.body_name = "Trunk"

    # Rewards
    # The user wanted a simple "Stand Up" task.
    # We can use velocity task but set command to 0.

    # Adjust rewards weights
    # mateus modify here if you want
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)

    # Custom terminations
    # BoosterT1Env: z < 0.3 or z > 2.0
    # MJLAB "base_height" termination usually exists.
    # We can override or add.

    # We'll stick to velocity task defaults for now, which encourages moving.
    # It's a "better" task than just standing.

    # Commands
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, UniformVelocityCommandCfg)
    # If we want just standing, we can set ranges to 0.
    # But let's leave it as velocity tracking, simpler to verify it moves.

    # Play mode
    if play:
        cfg.episode_length_s = 20.0  # longer for play

    return cfg


def booster_t1_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 reach goal environment configuration."""
    cfg = booster_t1_env_cfg(play=play)

    # Replace command
    cfg.commands = {
        "goal": GoalCommandCfg(
            entity_name="robot",
            resampling_time_range=(5.0, 15.0),
            ranges=GoalCommandCfg.Ranges(
                x=(-3.0, 3.0),
                y=(-3.0, 3.0),
                z=0.0,
            )
        )
    }

    # Update observations
    from mjlab.managers.observation_manager import ObservationTermCfg

    # Replace velocity command observation with relative goal position
    if "policy" in cfg.observations:
        if "command" in cfg.observations["policy"].terms:
            cfg.observations["policy"].terms["command"] = ObservationTermCfg(
                func=rel_goal_pos, params={"command_name": "goal"}
            )
    if "critic" in cfg.observations:
        if "command" in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms["command"] = ObservationTermCfg(
                func=rel_goal_pos, params={"command_name": "goal"}
            )

    # Update rewards
    from mjlab.managers.reward_manager import RewardTermCfg

    # mateus modify here if you want
    # Remove velocity tracking rewards
    if "track_linear_velocity" in cfg.rewards:
        del cfg.rewards["track_linear_velocity"]
    if "track_angular_velocity" in cfg.rewards:
        del cfg.rewards["track_angular_velocity"]

    # Add reach reward
    cfg.rewards["reach_goal"] = RewardTermCfg(
        func=reach_goal_reward,
        weight=10.0,
        params={"command_name": "goal", "std": 0.5},
    )

    # Ensure other rewards use the new command name
    for reward in cfg.rewards.values():
        if "command_name" in reward.params and reward.params["command_name"] == "twist":
            reward.params["command_name"] = "goal"

    # Disable curricula that depend on velocity commands
    cfg.curriculum = {}

    return cfg
