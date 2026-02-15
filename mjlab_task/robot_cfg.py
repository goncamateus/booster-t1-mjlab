import os
from dataclasses import dataclass, field
from mjlab.entity.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import BuiltinPositionActuatorCfg
from mujoco import MjSpec


@dataclass
class BoosterT1Cfg(EntityCfg):
    """Configuration for the Booster T1 robot."""

    # Configure articulation using default_factory and tuples for tyro compatibility
    articulation: EntityArticulationInfoCfg = field(
        default_factory=lambda: EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    # Match all internal joint but exclude the 'root' freejoint
                    target_names_expr=("(?!root$).*",),
                    stiffness=75.0,
                    damping=5.0,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        )
    )

    def __post_init__(self):
        # Configure MuJoCo model loading
        def get_booster_spec():
            model_path = os.path.join(
                os.path.dirname(__file__), "../assets/booster_t1/t1.xml"
            )
            spec = MjSpec.from_file(model_path)
            # Clear existing actuators to avoid collisions when mjlab adds its own
            while len(spec.actuators) > 0:
                spec.delete(spec.actuators[0])

            return spec

        self.spec_fn = get_booster_spec

        # Initial state from keyframe "home"
        self.init_state = EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.665),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "AAHead_yaw": 0.0,
                "Head_pitch": 0.0,
                "Left_Shoulder_Pitch": 0.0,
                "Left_Shoulder_Roll": -1.4,
                "Left_Elbow_Pitch": 0.0,
                "Left_Elbow_Yaw": -0.4,
                "Right_Shoulder_Pitch": 0.0,
                "Right_Shoulder_Roll": 1.4,
                "Right_Elbow_Pitch": 0.0,
                "Right_Elbow_Yaw": 0.4,
                "Waist": 0.0,
                "Left_Hip_Pitch": -0.2,
                "Left_Hip_Roll": 0.0,
                "Left_Hip_Yaw": 0.0,
                "Left_Knee_Pitch": 0.4,
                "Left_Ankle_Pitch": -0.2,
                "Left_Ankle_Roll": 0.0,
                "Right_Hip_Pitch": -0.2,
                "Right_Hip_Roll": 0.0,
                "Right_Hip_Yaw": 0.0,
                "Right_Knee_Pitch": 0.4,
                "Right_Ankle_Pitch": -0.2,
                "Right_Ankle_Roll": 0.0,
            },
        )
