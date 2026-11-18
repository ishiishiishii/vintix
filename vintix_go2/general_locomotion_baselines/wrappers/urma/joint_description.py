"""URMA joint / foot description vector layout (from unitree_go2/environment.py).

No MuJoCo / RL-X required: sizes are derived from the concatenate order in
`get_name_to_description_vector` so we can assert Go2 observation length offline.
"""

from __future__ import annotations

from dataclasses import dataclass

# Per-joint description = concat([...]) in environment.py (~L397-412)
JOINT_DESC_LAYOUT = (
    ("rel_pos_norm", 3),
    ("rel_axis", 3),
    ("nr_direct_child", 1),
    ("nominal_pos", 1),
    ("torque_limit", 1),
    ("joint_velocity_limit", 1),
    ("damping", 1),
    ("armature", 1),
    ("stiffness", 1),
    ("frictionloss", 1),
    ("joint_range", 2),  # jnt_range[i] is (low, high)
    ("pd_and_scale", 3),
    ("mass", 1),
    ("robot_dimensions", 3),
)
JOINT_DESCRIPTION_SIZE = sum(n for _, n in JOINT_DESC_LAYOUT)  # 23
# + position / velocity / previous_action
JOINT_DYNAMIC_EXTRA = 3
SINGLE_JOINT_OBS = JOINT_DESCRIPTION_SIZE + JOINT_DYNAMIC_EXTRA  # 26

# Per-foot description = concat([...]) (~L383-388)
FOOT_DESC_LAYOUT = (
    ("rel_pos_norm", 3),
    ("pd_and_scale", 3),
    ("mass", 1),
    ("robot_dimensions", 3),
)
FOOT_DESCRIPTION_SIZE = sum(n for _, n in FOOT_DESC_LAYOUT)  # 10
FOOT_DYNAMIC_EXTRA = 2  # ground_contact, cycles_since_last_ground_contact
SINGLE_FOOT_OBS = FOOT_DESCRIPTION_SIZE + FOOT_DYNAMIC_EXTRA  # 12

# General observations after dynamic blocks
GENERAL_OBS_LAYOUT = (
    ("trunk_lin_vel", 3),
    ("trunk_ang_vel", 3),
    ("goal_vel", 3),
    ("projected_gravity", 3),
    ("height", 1),
    ("pd_and_scale", 3),
    ("mass", 1),
    ("robot_dimensions", 3),
)
GENERAL_OBS_SIZE = sum(n for _, n in GENERAL_OBS_LAYOUT)  # 20

GO2_NUM_JOINTS = 12
GO2_NUM_FEET = 4
GO2_ACTION_DIM = 12

GO2_OBS_DIM = (
    GO2_NUM_JOINTS * SINGLE_JOINT_OBS
    + GO2_NUM_FEET * SINGLE_FOOT_OBS
    + GENERAL_OBS_SIZE
)  # 312 + 48 + 20 = 380


@dataclass(frozen=True)
class UrmaGo2Dims:
    joint_description: int = JOINT_DESCRIPTION_SIZE
    foot_description: int = FOOT_DESCRIPTION_SIZE
    single_joint_obs: int = SINGLE_JOINT_OBS
    single_foot_obs: int = SINGLE_FOOT_OBS
    general_obs: int = GENERAL_OBS_SIZE
    num_joints: int = GO2_NUM_JOINTS
    num_feet: int = GO2_NUM_FEET
    obs_dim: int = GO2_OBS_DIM
    action_dim: int = GO2_ACTION_DIM
