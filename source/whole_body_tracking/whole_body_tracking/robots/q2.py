import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# 00: 低速端等效惯量0.001kgm2 01/02：低速端等效惯量4.2e-3 kg*m2    03：低速端等效惯量0.02kgm2  04：低速端等效惯量0.04 kg*m2  05：低速端等效惯量0.0007kgm2  06：低速端等效惯量0.012kgm2

ROBSTRIDE_04 = 0.04
ROBSTRIDE_03 = 0.02
ROBSTRIDE_02 = 0.0042

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_04 = ROBSTRIDE_04 * NATURAL_FREQ**2
STIFFNESS_03 = ROBSTRIDE_03 * NATURAL_FREQ**2
STIFFNESS_02 = ROBSTRIDE_02 * NATURAL_FREQ**2


DAMPING_04 = 2.0 * DAMPING_RATIO * ROBSTRIDE_04 * NATURAL_FREQ
DAMPING_03 = 2.0 * DAMPING_RATIO * ROBSTRIDE_03 * NATURAL_FREQ
DAMPING_02 = 2.0 * DAMPING_RATIO * ROBSTRIDE_02 * NATURAL_FREQ



Q2_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/q2_19dof/q2_19dof.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.331614,
            ".*_knee_joint": 0.5236,
            ".*_ankle_pitch_joint": -0.191986,
            ".*_elbow_joint": 0.0,   #0.6
            "left_shoulder_roll_joint": 0.0,   #0.2
            "left_shoulder_pitch_joint": 0.0,  #0.2
            "right_shoulder_roll_joint": -0.0, #-0.2
            "right_shoulder_pitch_joint": 0.0,  #0.2
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_pitch_joint": 120.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 30.0,
                ".*_hip_roll_joint": 30.0,
                ".*_hip_pitch_joint": 30.0,
                ".*_knee_joint": 30.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_04,
                ".*_hip_roll_joint": STIFFNESS_04,
                ".*_hip_yaw_joint": STIFFNESS_04,
                ".*_knee_joint": STIFFNESS_04,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_04,
                ".*_hip_roll_joint": DAMPING_04,
                ".*_hip_yaw_joint": DAMPING_04,
                ".*_knee_joint": DAMPING_04,
            },
            armature={
                ".*_hip_pitch_joint": ROBSTRIDE_04,
                ".*_hip_roll_joint": ROBSTRIDE_04,
                ".*_hip_yaw_joint": ROBSTRIDE_04,
                ".*_knee_joint": ROBSTRIDE_04,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=17.0,
            velocity_limit_sim=60.0,
            joint_names_expr=[".*_ankle_pitch_joint"],
            stiffness=2.0 * STIFFNESS_02,
            damping=2.0 * DAMPING_02,
            armature=2.0 * ROBSTRIDE_02,
        ),
        # "waist": ImplicitActuatorCfg(
        #     effort_limit_sim=50,
        #     velocity_limit_sim=37.0,
        #     joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        #     stiffness=2.0 * STIFFNESS_02,
        #     damping=2.0 * DAMPING_02,
        #     armature=2.0 * ROBSTRIDE_02,
        # ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=120,
            velocity_limit_sim=30.0,
            joint_names_expr=["torso_joint"],
            stiffness=STIFFNESS_04,
            damping=DAMPING_04,
            armature=ROBSTRIDE_04,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",

            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 60.0,
                ".*_shoulder_roll_joint": 17.0,
                ".*_shoulder_yaw_joint": 17.0,
                ".*_elbow_joint": 17.0,

            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 30.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 60.0,
                ".*_elbow_joint": 60.0,

            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_03,
                ".*_shoulder_roll_joint": STIFFNESS_02,
                ".*_shoulder_yaw_joint": STIFFNESS_02,
                ".*_elbow_joint": STIFFNESS_02,

            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_03,
                ".*_shoulder_roll_joint": DAMPING_02,
                ".*_shoulder_yaw_joint": DAMPING_02,
                ".*_elbow_joint": DAMPING_02,

            },
            armature={
                ".*_shoulder_pitch_joint": ROBSTRIDE_03,
                ".*_shoulder_roll_joint": ROBSTRIDE_02,
                ".*_shoulder_yaw_joint": ROBSTRIDE_02,
                ".*_elbow_joint": ROBSTRIDE_02,

            },
        ),
    },
)

Q2_ACTION_SCALE = {}
for a in Q2_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            Q2_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
