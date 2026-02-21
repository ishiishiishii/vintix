import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from env import Go2Env
from env import MiniCheetahEnv
from env import LaikagoEnv
from env import UnitreeA1Env
from env import ANYmalCEnv
from spotmicro_env import SpotMicroEnv
from env import Go1Env



def get_train_cfg(exp_name, max_iterations, seed=1):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,  # 100イテレーションごとに保存
        "empirical_normalization": None,
        "seed": seed,
    }

    return train_cfg_dict


def get_go2_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def get_minicheetah_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "torso_to_abduct_fl_j": 0.0,
            "torso_to_abduct_fr_j": 0.0,
            "torso_to_abduct_hr_j": 0.0,
            "torso_to_abduct_hl_j": 0.0,
            "abduct_fl_to_thigh_fl_j": -0.8,
            "abduct_fr_to_thigh_fr_j": -0.8,
            "abduct_hr_to_thigh_hr_j": -0.8,
            "abduct_hl_to_thigh_hl_j": -0.8,
            "thigh_fl_to_knee_fl_j": 1.5,
            "thigh_fr_to_knee_fr_j": 1.5,
            "thigh_hr_to_knee_hr_j": 1.5,
            "thigh_hl_to_knee_hl_j": 1.5,
        },
        "joint_names": [
            "torso_to_abduct_fr_j",
            "abduct_fr_to_thigh_fr_j",
            "thigh_fr_to_knee_fr_j",
            "torso_to_abduct_fl_j",
            "abduct_fl_to_thigh_fl_j",
            "thigh_fl_to_knee_fl_j",
            "torso_to_abduct_hr_j",
            "abduct_hr_to_thigh_hr_j",
            "thigh_hr_to_knee_hr_j",
            "torso_to_abduct_hl_j",
            "abduct_hl_to_thigh_hl_j",
            "thigh_hl_to_knee_hl_j",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        # Minicheetahの脚の全長は約0.418m（URDFから計算）
        # 関節を曲げたスタンディング姿勢では、実際の高さは約0.30m程度
        # 初期高さを0.40mに設定することで、ロボットが空中から少し落ちるような挙動になる
        "base_init_pos": [0.0, 0.0, 0.40],  # 目標高さ0.30mより高い初期位置（空中から落下する挙動）
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        # Minicheetahの実際のスタンディング高さに基づく設定
        # 初期関節角度での実際の高さは0.30m（測定済み）
        # 目標高さを0.30mに設定し、初期位置を0.40mにして空中から落下する挙動にする
        # Go2と同じ報酬設定に変更（minicheetah-walking3用）
        # 以前の設定（コメント）:
        # "base_height_target": 0.30,  # 初期関節角度での実際の高さ（30cm）に基づく
        # "feet_height_target": 0.06,  # 足の目標高さ（6cm）
        "base_height_target": 0.3,  # Go2と同じ設定
        "feet_height_target": 0.075,  # Go2と同じ設定
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def get_laikago_cfgs():
    """
    Laikago専用の設定（URDFと物理的特性に基づく）
    Laikagoの仕様:
    - 高さ: 60cm（立ち姿勢）
    - 重量: 約22kg
    - ベース質量: 13.715kg（URDFから）
    - 最大歩行速度: 0.8m/秒
    """
    env_cfg = {
        "num_actions": 12,
        # joint/link names (URDFから確認済み)
        "default_joint_angles": {  # [rad] - Laikagoの適切なスタンディング姿勢
            # Front Right leg (FR) - 1st
            "FR_hip_motor_2_chassis_joint": 0.0,      # Chassis: 0
            "FR_upper_leg_2_hip_motor_joint": 0.0,    # Upperleg: 0
            "FR_lower_leg_2_upper_leg_joint": -0.75,  # Lowerleg: -0.75
            # Front Left leg (FL) - 2nd
            "FL_hip_motor_2_chassis_joint": 0.0,      # Chassis: 0
            "FL_upper_leg_2_hip_motor_joint": 0.0,   # Upperleg: 0
            "FL_lower_leg_2_upper_leg_joint": -0.75, # Lowerleg: -0.75
            # Rear Right leg (RR) - 3rd
            "RR_hip_motor_2_chassis_joint": 0.0,      # Chassis: 0
            "RR_upper_leg_2_hip_motor_joint": 0.0,   # Upperleg: 0
            "RR_lower_leg_2_upper_leg_joint": -0.75, # Lowerleg: -0.75
            # Rear Left leg (RL) - 4th
            "RL_hip_motor_2_chassis_joint": 0.0,      # Chassis: 0
            "RL_upper_leg_2_hip_motor_joint": 0.0,   # Upperleg: 0
            "RL_lower_leg_2_upper_leg_joint": -0.75  # Lowerleg: -0.75
        },
        "joint_names": [
            # Front Right leg (FR) - 1st
            "FR_hip_motor_2_chassis_joint",
            "FR_upper_leg_2_hip_motor_joint", 
            "FR_lower_leg_2_upper_leg_joint",
            # Front Left leg (FL) - 2nd
            "FL_hip_motor_2_chassis_joint",
            "FL_upper_leg_2_hip_motor_joint",
            "FL_lower_leg_2_upper_leg_joint", 
            # Rear Right leg (RR) - 3rd
            "RR_hip_motor_2_chassis_joint",
            "RR_upper_leg_2_hip_motor_joint",
            "RR_lower_leg_2_upper_leg_joint",
            # Rear Left leg (RL) - 4th
            "RL_hip_motor_2_chassis_joint", 
            "RL_upper_leg_2_hip_motor_joint",
            "RL_lower_leg_2_upper_leg_joint"
        ],
        # PD control parameters (Laikagoに適した値)
        "kp": 20.0,  # 位置ゲイン
        "kd": 0.5,   # 速度ゲイン
        # termination conditions
        "termination_if_roll_greater_than": 10,   # degree
        "termination_if_pitch_greater_than": 10,  # degree
        # base pose (Laikagoの立ち姿勢高さ60cmを考慮、初期高さを50cmに設定)
        # 初期状態（回転なし）: 正面=Z軸、足=-Y軸 → 目標状態: 正面=X軸、足=-Z軸
        # 固定座標系での回転: Y軸90度→X軸90度（R_x @ R_y の順序）
        # Genesis座標系: X=前後（前進方向）、Y=左右（横方向）、Z=上下（高さ方向）
        # transform_quat_by_quat(v, u)は R_u @ R_v を計算するため、固定座標系での回転
        "base_init_pos": [0.0, 0.0, 0.50],  # Laikagoのスタンディング高さ（50cm）
        "base_init_quat": [0.5, 0.5, 0.5, 0.5],  # Y軸90度→X軸90度（固定座標系、正面=X軸、足=-Z軸）
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,  # アクションスケール
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,  # 3(ang_vel) + 3(gravity) + 3(commands) + 12(dof_pos) + 12(dof_vel) + 12(actions)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.45,  # Laikagoの目標ベース高さ（45cm）
        "feet_height_target": 0.095,  # 足の目標高さ
        "reward_scales": {
            "tracking_lin_vel": 1.0,      # 線形速度追従
            "tracking_ang_vel": 0.2,      # 角速度追従
            "lin_vel_z": -1.0,            # Z方向速度ペナルティ
            "base_height": -50.0,         # ベース高さペナルティ
            "action_rate": -0.005,       # アクション変化率ペナルティ
            "similar_to_default": -0.1,  # デフォルト姿勢からの乖離ペナルティ
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],  # 前進速度コマンド（最大0.8m/秒だが、訓練時は0.5m/秒）
        "lin_vel_y_range": [0, 0],      # 横方向速度コマンド
        "ang_vel_range": [0, 0],         # 角速度コマンド
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def get_unitreea1_cfgs():
    """
    UnitreeA1の最適化された設定
    URDFファイルから取得した情報に基づいて最適化:
    - ジョイントリミット: hip: ±0.80 rad, thigh: -1.05 to 4.19 rad, calf: -2.70 to -0.92 rad
    - Trunk質量: 6.0 kg
    - 適切なスタンディング姿勢を設定（Go2とMiniCheetahの設定を参考に最適化）
    """
    env_cfg = {
        "num_actions": 12,
        # joint/link names (URDFから確認済み)
        "default_joint_angles": {  # [rad] - UnitreeA1の適切なスタンディング姿勢（Go2の設定を参考に修正）
            # 前脚: やや低めの姿勢で安定性を確保
            "FR_hip_joint": 0.0,      # 中間位置（リミット: ±0.80）
            "FR_thigh_joint": 0.8,    # 約46度前向き（リミット: -1.05 to 4.19）- Go2と同じ0.8に修正
            "FR_calf_joint": -1.5,    # 約-86度（リミット: -2.70 to -0.92）- Go2の-1.5に修正
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,    # Go2と同じ0.8に修正
            "FL_calf_joint": -1.5,    # Go2の-1.5に修正
            # 後脚: 前脚よりやや高めで推進力を確保
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 1.0,    # 約57度（リミット: -1.05 to 4.19）- Go2の1.0と同じ
            "RR_calf_joint": -1.5,    # Go2の-1.5に修正
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 1.0,    # Go2の1.0と同じ
            "RL_calf_joint": -1.5,    # Go2の-1.5に修正
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD control parameters (UnitreeA1に適した値 - Go2と同じ)
        "kp": 20.0,  # 位置ゲイン
        "kd": 0.5,   # 速度ゲイン
        # termination conditions
        "termination_if_roll_greater_than": 10,   # degree
        "termination_if_pitch_greater_than": 10,  # degree
        # base pose (URDFから計算された適切な初期高さ)
        "base_init_pos": [0.0, 0.0, 0.42],  # UnitreeA1のスタンディング高さ（30cm）- URDF解析結果に基づく
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,  # アクションスケール
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,  # 3(ang_vel) + 3(gravity) + 3(commands) + 12(dof_pos) + 12(dof_vel) + 12(actions)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.27,  # UnitreeA1の目標ベース高さ（27cm）- URDF解析結果に基づく
        "feet_height_target": 0.075,  # 足の目標高さ - Go2と同じ
        "reward_scales": {
            "tracking_lin_vel": 1.0,      # 線形速度追従
            "tracking_ang_vel": 0.2,      # 角速度追従
            "lin_vel_z": -1.0,            # Z方向速度ペナルティ
            "base_height": -50.0,         # ベース高さペナルティ
            "action_rate": -0.005,       # アクション変化率ペナルティ
            "similar_to_default": -0.1,  # デフォルト姿勢からの乖離ペナルティ
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],  # 前進速度コマンド
        "lin_vel_y_range": [0, 0],      # 横方向速度コマンド
        "ang_vel_range": [0, 0],         # 角速度コマンド
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def get_go1_cfgs():
    """
    UnitreeGo1の最適化された設定
    URDFファイルから取得した情報に基づいて最適化:
    - ジョイントリミット: hip: -0.863 to 0.863 rad, thigh: -0.686 to 4.501 rad, calf: -2.818 to -0.888 rad
    - Trunk質量: 5.204 kg (Go2: 6.921kgより軽い)
    - Trunkサイズ: 0.3762m (長さ) x 0.0935m (幅) x 0.114m (高さ) - Go2と同じ
    - 脚の全長: 0.506m (hip: 0.08m + thigh: 0.213m + calf: 0.213m) - Go2の0.521mより0.015m短い
    - デフォルトジョイント角度での実効的な脚の高さ: 約0.446m (Go2: 0.461m)
    - スタンディング時のベース高さ: 約0.50m (脚0.446m + trunk高さ/2 0.057m) - Go2の0.518mより0.015m低い
    - Go2との比率: 0.971 (Go1/Go2)
    - Go2の設定比率を維持: base_height_target/base_init_pos = 0.30/0.42 = 0.714
    """
    env_cfg = {
        "num_actions": 12,
        # joint/link names (URDFから確認済み)
        "default_joint_angles": {  # [rad] - UnitreeGo1の適切なスタンディング姿勢
            # 前脚: やや低めの姿勢で安定性を確保
            "FR_hip_joint": 0.0,      # 中間位置（リミット: -0.863 to 0.863）
            "FR_thigh_joint": 0.8,    # 約46度前向き（リミット: -0.686 to 4.501）
            "FR_calf_joint": -1.6,    # 約-92度（リミット: -2.818 to -0.888）
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.6,
            # 後脚: 前脚よりやや高めで推進力を確保
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 1.0,    # 約57度（リミット: -0.686 to 4.501）
            "RR_calf_joint": -1.6,    # 約-92度
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.6,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD control parameters (UnitreeGo1に適した値 - Go2と同じ)
        "kp": 20.0,  # 位置ゲイン
        "kd": 0.5,   # 速度ゲイン
        # termination conditions
        "termination_if_roll_greater_than": 10,   # degree
        "termination_if_pitch_greater_than": 10,  # degree
        # base pose: 初期高さを0.4mに設定
        "base_init_pos": [0.0, 0.0, 0.42],  # 初期高さ0.4m（40cm）に設定
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,  # アクションスケール
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,  # 3(ang_vel) + 3(gravity) + 3(commands) + 12(dof_pos) + 12(dof_vel) + 12(actions)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        # 目標高さを0.32m（32cm）に設定
        "base_height_target": 0.32, # 目標高さ0.32m（32cm）に設定
        # Go2の0.075mを比率0.971で調整: 0.075 * 0.971 = 0.073m
        "feet_height_target": 0.075,  # Go2の0.075mを比率0.971で調整（Go2: 0.075m）
        "reward_scales": {
            "tracking_lin_vel": 1.0,      # 線形速度追従
            "tracking_ang_vel": 0.2,      # 角速度追従
            "lin_vel_z": -1.0,            # Z方向速度ペナルティ
            "base_height": -50.0,         # ベース高さペナルティ
            "action_rate": -0.005,       # アクション変化率ペナルティ
            "similar_to_default": -0.1,  # デフォルト姿勢からの乖離ペナルティ
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],  # 前進速度コマンド
        "lin_vel_y_range": [0, 0],      # 横方向速度コマンド
        "ang_vel_range": [0, 0],         # 角速度コマンド
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def get_anymalc_cfgs():
    """
    ANYmalC専用の設定（URDFと物理的特性に基づく）
    - URDF解析結果に基づいて正しい初期姿勢を設定
    - 脚の全長: 約84cm（太もも0.169m + すね0.338m + 足0.338m）
    - HFE: 前脚0.5 rad（約29度）、後脚0.7 rad（約40度）
    - KFE: すべて-1.2 rad（約-69度）
    """
    env_cfg = {
        "num_actions": 12,
        # joint/link names (URDFから確認済み)
        "default_joint_angles": {  # [rad] - URDF表示に基づく適切なスタンディング姿勢
            # 左前脚 (LF - Left Front)
            "LF_HAA": -0.3,     # Hip Abduction/Adduction
            "LF_HFE": 0.8,      # Hip Flexion/Extension
            "LF_KFE": -1.3,     # Knee Flexion/Extension
            # 右前脚 (RF - Right Front)
            "RF_HAA": 0.3,      # Hip Abduction/Adduction
            "RF_HFE": 0.8,      # Hip Flexion/Extension
            "RF_KFE": -1.3,     # Knee Flexion/Extension
            # 左後脚 (LH - Left Hind)
            "LH_HAA": -0.3,     # Hip Abduction/Adduction
            "LH_HFE": -0.8,     # Hip Flexion/Extension
            "LH_KFE": 1.3,      # Knee Flexion/Extension
            # 右後脚 (RH - Right Hind)
            "RH_HAA": 0.3,      # Hip Abduction/Adduction
            "RH_HFE": -0.8,     # Hip Flexion/Extension
            "RH_KFE": 1.3,      # Knee Flexion/Extension
        },
        "joint_names": [
            "RH_HAA",
            "LH_HAA",
            "RF_HAA",
            "LF_HAA",
            "RH_HFE",
            "LH_HFE",
            "RF_HFE",
            "LF_HFE",
            "RH_KFE",
            "LH_KFE",
            "RF_KFE",
            "LF_KFE",
        ],
        # PD control parameters
        "kp": 20.0,  # 位置ゲイン
        "kd": 0.5,   # 速度ゲイン
        # termination conditions
        "termination_if_roll_greater_than": 10,   # degree
        "termination_if_pitch_greater_than": 10,  # degree
        # base pose（脚の全長84cmを考慮、KFE=-0.6 radに合わせて調整）
        "base_init_pos": [0.0, 0.0, 0.55],  # 初期高さ
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,  # Go2と同じ
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,  # 3(ang_vel) + 3(gravity) + 3(commands) + 12(dof_pos) + 12(dof_vel) + 12(actions)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,   # Go2と同じ
        "feet_height_target": 0.075,  # Go2と同じ
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.9, 0.9],  # 前進速度コマンド（0.9 m/sに変更）
        "lin_vel_y_range": [0, 0],      # 横方向速度コマンド
        "ang_vel_range": [0, 0],         # 角速度コマンド
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def get_spotmicro_cfgs():
    """
    SpotMicro専用の設定（URDFと物理的特性に基づく、Go2と同じ形式）
    - URDF解析結果:
      * ベースサイズ: 14cm x 11cm x 7cm
      * ベース質量: 1.20 kg
      * 脚の全長: 約28.7cm（shoulder offset 5.2cm + leg 12cm + foot 11.5cm）
    - Go2との比較:
      * Go2ベース: 37.6cm x 9.35cm x 11.4cm
      * Go2初期位置: 42cm、目標高さ: 30cm
      * 脚の長さ比: 28.7cm / 52.1cm ≈ 0.551
      * 推奨初期位置: 42cm × 0.551 = 23.1cm
      * 推奨目標高さ: 30cm × 0.551 = 16.5cm (Go2の比率0.714を維持)
    """
    env_cfg = {
        "num_actions": 12,
        # joint/link names (URDFから確認済み)
        "default_joint_angles": {  # [rad] - SpotMicroの適切なスタンディング姿勢
            # 前左脚 (FL - Front Left)
            "motor_front_left_shoulder": 0.0,      # 肩関節（横方向の中立位置、リミット: ±1.0 rad）
            "motor_front_left_leg": -0.8,         # 脚関節（約-46度、リミット: -2.17 to 0.97 rad）
            "foot_motor_front_left": 1.5,         # 足関節（約86度、リミット: -0.1 to 2.59 rad）
            # 前右脚 (FR - Front Right)
            "motor_front_right_shoulder": 0.0,     # 肩関節（横方向の中立位置）
            "motor_front_right_leg": -0.8,        # 脚関節（約-46度）
            "foot_motor_front_right": 1.5,        # 足関節（約86度）
            # 後左脚 (RL - Rear Left)
            "motor_rear_left_shoulder": 0.0,       # 肩関節（横方向の中立位置）
            "motor_rear_left_leg": -0.8,           # 脚関節（約-46度）
            "foot_motor_rear_left": 1.5,           # 足関節（約86度）
            # 後右脚 (RR - Rear Right)
            "motor_rear_right_shoulder": 0.0,      # 肩関節（横方向の中立位置）
            "motor_rear_right_leg": -0.8,         # 脚関節（約-46度）
            "foot_motor_rear_right": 1.5,         # 足関節（約86度）
        },
        "joint_names": [
            "motor_front_right_shoulder",
            "motor_front_right_leg",
            "foot_motor_front_right",
            "motor_front_left_shoulder",
            "motor_front_left_leg",
            "foot_motor_front_left",
            "motor_rear_right_shoulder",
            "motor_rear_right_leg",
            "foot_motor_rear_right",
            "motor_rear_left_shoulder",
            "motor_rear_left_leg",
            "foot_motor_rear_left",
        ],
        # PD control parameters (Go2と同じ)
        "kp": 20.0,
        "kd": 0.5,
        # termination conditions (Go2と同じ)
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        "termination_if_base_height_below": 0.08,  # ベース高さが8cm以下になったら転倒とみなす（目標高さ15.5cmの約半分）
        # base pose (SpotMicroの実際のスタンディング高さに基づく)
        "base_init_pos": [0.0, 0.0, 0.25],  # 初期高さ（25cm）
        "base_init_quat": [0.0, 0.0, 0.0, 1.0],  # Z軸周りに180度回転（前後を入れ替える）
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.2,  # 目標ベース高さ（20cm）
        "feet_height_target": 0.041,  # 足の目標高さ（4.1cm）- Go2の7.5cm × 0.551、比率0.179を維持
        "reward_scales": {
            "tracking_lin_vel": 10.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.5,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.4, 0.4],  # SpotMicroの目標線形速度（40cm/s）
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago", "unitreea1", "anymalc", "go1", "spotmicro"], 
                        default="go2", help="Robot type to train")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=301)
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained model for fine-tuning")
    
    # ドメインランダマイゼーション関連の引数
    parser.add_argument("--domain_randomization", action="store_true", default=False,
                        help="Enable domain randomization")
    parser.add_argument("--mass_range_min", type=float, default=0.9,
                        help="Minimum mass scale for domain randomization")
    parser.add_argument("--mass_range_max", type=float, default=1.1,
                        help="Maximum mass scale for domain randomization")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()

    gs.init(logging_level="warning",precision="64") #precision="64"を加えた

    log_dir = f"../../logs/{args.exp_name}"
    # ロボットタイプに応じて設定関数を選択
    if args.robot_type == "go2":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_go2_cfgs()
    elif args.robot_type == "minicheetah":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_minicheetah_cfgs()
    elif args.robot_type == "laikago":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_laikago_cfgs()
    elif args.robot_type == "unitreea1":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_unitreea1_cfgs()
    elif args.robot_type == "anymalc":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_anymalc_cfgs()
    elif args.robot_type == "go1":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_go1_cfgs()
    elif args.robot_type == "spotmicro":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_spotmicro_cfgs()
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")
    
    # ドメインランダマイゼーション設定を取得
    dr_cfg = {
        "domain_randomization": args.domain_randomization,
        "mass_range": (args.mass_range_min, args.mass_range_max),
    }
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # 既存のログディレクトリがある場合、resumeする場合は削除しない
    # --pretrained_pathが指定されていて、同じディレクトリから続きを訓練する場合は削除しない
    resume_existing = args.pretrained_path is None and os.path.exists(log_dir) and os.path.exists(os.path.join(log_dir, "cfgs.pkl"))
    use_pretrained_in_same_dir = args.pretrained_path is not None and os.path.exists(log_dir) and os.path.exists(os.path.join(log_dir, "cfgs.pkl"))
    
    if not resume_existing and not use_pretrained_in_same_dir:
        # 新規訓練の場合のみログディレクトリを削除
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )
    elif resume_existing or use_pretrained_in_same_dir:
        # resumeする場合、または--pretrained_pathで同じディレクトリから続きを訓練する場合は既存の設定を読み込む
        if use_pretrained_in_same_dir:
            print(f"Continuing training in existing log directory: {log_dir}")
        else:
            print(f"Resuming training from existing log directory: {log_dir}")
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
        train_cfg["runner"]["max_iterations"] = args.max_iterations

    # ロボットタイプに応じて環境を作成（ドメインランダマイゼーションパラメータを含む）
    if args.robot_type == "go2":
        env = Go2Env(
            num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"]
        )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"]
        )
    elif args.robot_type == "laikago":
        env = LaikagoEnv(
            num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"]
        )
    elif args.robot_type == "unitreea1":
        env = UnitreeA1Env(
            num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"]
        )
    elif args.robot_type == "anymalc":
        env = ANYmalCEnv(
            num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"]
        )
    elif args.robot_type == "go1":
        env = Go1Env(
            num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"]
        )
    elif args.robot_type == "spotmicro":
        env = SpotMicroEnv(
            num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"]
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")

    # ドメインランダマイゼーション設定を表示
    if dr_cfg["domain_randomization"]:
        print(f"ドメインランダマイゼーション有効:")
        print(f"  質量範囲: {dr_cfg['mass_range'][0]:.2f} - {dr_cfg['mass_range'][1]:.2f}")
    else:
        print("ドメインランダマイゼーション無効")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    # ファインチューニングまたはresumeの判定とモデル読み込み
    if args.pretrained_path:
        if os.path.exists(args.pretrained_path):
            print(f"Fine-tuning mode: Loading pretrained model from {args.pretrained_path}")
            runner.load(args.pretrained_path)
        else:
            print(f"Warning: Pretrained model not found at {args.pretrained_path}")
            print("Starting training from scratch...")
    elif resume_existing:
        # 既存のログディレクトリから最新のチェックポイントを探す
        import glob
        checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
        if checkpoints:
            # 最新のチェックポイントを取得（数字が最大のもの）
            latest_checkpoint = max(checkpoints, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            print(f"Resuming training from latest checkpoint: {latest_checkpoint}")
            runner.load(latest_checkpoint)
        else:
            print("No checkpoint found, starting training from scratch...")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# 通常の学習
python examples/locomotion/train.py -e test-go2 -r go2 --max_iterations 101

# ドメインランダマイゼーション有効での学習
python examples/locomotion/train.py -e test-go2-domain-randomized -r go2 --domain_randomization --mass_range_min 0.8 --mass_range_max 1.2 --max_iterations 101

# ファインチューニング
python examples/locomotion/train.py -e test-go2-finetuning -r go2 --pretrained_path logs/mini_cheetah-walking/model_300.pt --max_iterations 601 --seed 1

# ドメインランダマイゼーション有効でのファインチューニング
python examples/locomotion/train.py -e test-go2-dr-finetuning -r go2 --domain_randomization --mass_range_min 0.9 --mass_range_max 1.1 --pretrained_path logs/go2-walking/model_100.pt --max_iterations 21

# 異なるロボット間でのファインチューニング
python examples/locomotion/train.py -e test-minicheetah-to-go2 -r go2 --pretrained_path logs/minicheetah-walking/model_100.pt --max_iterations 31
"""
