from isaaclab.utils import configclass

from whole_body_tracking.robots.q2 import Q2_ACTION_SCALE, Q2_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.q2.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class Q2FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = Q2_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = Q2_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_pitch_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_pitch_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            # "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            # "right_wrist_yaw_link",
        ]


@configclass
class Q2FlatWoStateEstimationEnvCfg(Q2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class Q2FlatLowFreqEnvCfg(Q2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
