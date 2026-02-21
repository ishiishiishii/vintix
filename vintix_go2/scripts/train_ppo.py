import argparse
import os
import pickle
import shutil
import sys
from importlib import metadata
from pathlib import Path

# Genesis locomotion環境のインポート用にパスを追加
GENESIS_LOCOMOTION_PATH = str(Path(__file__).parents[2] / "Genesis" / "examples" / "locomotion")
sys.path.insert(0, GENESIS_LOCOMOTION_PATH)

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
from env import Go1Env
from spotmicro_env import SpotMicroEnv



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
        "save_interval": 10,
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
        "base_init_pos": [0.0, 0.0, 0.45], #0.45から変えた
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
        "base_height_target": 0.3, #0.3から変えた
        "feet_height_target": 0.075, #0.075から変えた
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
            "FR_hip_motor_2_chassis_joint": 0.0,
            "FR_upper_leg_2_hip_motor_joint": -0.8,  # 約-46度（前脚はやや低め）
            "FR_lower_leg_2_upper_leg_joint": 1.5,   # 約86度（膝を適度に曲げる）
            # Front Left leg (FL) - 2nd
            "FL_hip_motor_2_chassis_joint": 0.0,
            "FL_upper_leg_2_hip_motor_joint": -0.8,
            "FL_lower_leg_2_upper_leg_joint": 1.5,
            # Rear Right leg (RR) - 3rd
            "RR_hip_motor_2_chassis_joint": 0.0,
            "RR_upper_leg_2_hip_motor_joint": -0.8,
            "RR_lower_leg_2_upper_leg_joint": 1.5,
            # Rear Left leg (RL) - 4th
            "RL_hip_motor_2_chassis_joint": 0.0,
            "RL_upper_leg_2_hip_motor_joint": -0.8,
            "RL_lower_leg_2_upper_leg_joint": 1.5
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
        "base_init_pos": [0.0, 0.0, 0.50],  # Laikagoのスタンディング高さ（50cm）
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
        "base_height_target": 0.50,  # Laikagoの目標ベース高さ（50cm）- 立ち姿勢60cmを考慮
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
        "default_joint_angles": {  # [rad] - UnitreeA1の適切なスタンディング姿勢
            # 前脚: やや低めの姿勢で安定性を確保
            "FR_hip_joint": 0.0,      # 中間位置（リミット: ±0.80）
            "FR_thigh_joint": 0.9,    # 約52度前向き（リミット: -1.05 to 4.19）- Go2の0.8に近い
            "FR_calf_joint": -1.6,    # 約-92度（リミット: -2.70 to -0.92）- Go2の-1.5に近い
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.6,
            # 後脚: 前脚よりやや高めで推進力を確保
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 1.0,    # 約57度（リミット: -1.05 to 4.19）- Go2の1.0と同じ
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
        # PD control parameters (UnitreeA1に適した値 - Go2と同じ)
        "kp": 20.0,  # 位置ゲイン
        "kd": 0.5,   # 速度ゲイン
        # termination conditions
        "termination_if_roll_greater_than": 10,   # degree
        "termination_if_pitch_greater_than": 10,  # degree
        # base pose (UnitreeA1の適切な初期高さ - Go2の0.42とMiniCheetahの0.45の中間)
        "base_init_pos": [0.0, 0.0, 0.40],  # UnitreeA1のスタンディング高さ（40cm）
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
        "base_height_target": 0.32,  # UnitreeA1の目標ベース高さ（32cm）- Go2の0.3とMiniCheetahの0.3を参考
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
        # base pose: 初期高さを0.361mに設定（check_go1_height.pyで測定・調整済み）
        # 初期関節角度で足が地面に届くように調整された値
        "base_init_pos": [0.0, 0.0, 0.361],  # 初期高さ0.361m（36.1cm）に設定
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
        # Go2の設定（0.30m）をGo1との比率（0.971）で調整: 0.30 * 0.971 = 0.291m
        # Go1はGo2より小さいので、目標高さもGo2より低く設定（さらに低めに調整）
        "base_height_target": 0.32,  # 目標高さ0.32m（32cm）に設定  # Go2の0.30mを比率0.971で調整した0.291mよりさらに低く設定（Go2: 0.30m）
        # Go2の0.075mを比率0.971で調整: 0.075 * 0.971 = 0.073m
        "feet_height_target": 0.073,  # Go2の0.075mを比率0.971で調整（Go2: 0.075m）
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
        "default_joint_angles": {  # [rad] - SpotMicroの歩行姿勢（Go1を参考に調整）
            # 前左脚 (FL - Front Left)
            "motor_front_left_shoulder": 0.0,      # 肩関節（横方向の中立位置、リミット: ±1.0 rad）
            "motor_front_left_leg": 0.8,           # 脚関節（約46度、リミット: -2.17 to 0.97 rad）- 正の値で脚を下に伸ばす
            "foot_motor_front_left": 1.2,          # 足関節（約69度、リミット: -0.1 to 2.59 rad）- 正の値で足を下に向ける（Go1のcalf_joint: -1.6を参考）
            # 前右脚 (FR - Front Right)
            "motor_front_right_shoulder": 0.0,     # 肩関節（横方向の中立位置）
            "motor_front_right_leg": 0.8,         # 脚関節（約46度）
            "foot_motor_front_right": 1.2,         # 足関節（約69度）
            # 後左脚 (RL - Rear Left)
            "motor_rear_left_shoulder": 0.0,       # 肩関節（横方向の中立位置）
            "motor_rear_left_leg": 1.0,            # 脚関節（約57度、Go1の後脚thigh_joint: 1.0を参考）- 後脚はやや高め
            "foot_motor_rear_left": 1.2,           # 足関節（約69度）
            # 後右脚 (RR - Rear Right)
            "motor_rear_right_shoulder": 0.0,     # 肩関節（横方向の中立位置）
            "motor_rear_right_leg": 1.0,          # 脚関節（約57度）- 後脚はやや高め
            "foot_motor_rear_right": 1.2,         # 足関節（約69度）
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
        # base pose (Go2の比率を維持: 0.714 = base_height_target / base_init_pos)
        "base_init_pos": [0.0, 0.0, 0.23],  # 初期高さ（23cm）- Go2の42cm × 0.551
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
        "base_height_target": 0.165,  # 目標ベース高さ（16.5cm）- Go2の30cm × 0.551、比率0.714を維持
        "feet_height_target": 0.041,  # 足の目標高さ（4.1cm）- Go2の7.5cm × 0.551、比率0.179を維持
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago", "unitreea1", "go1", "spotmicro"], 
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

    log_dir = f"logs/{args.exp_name}"
    # ロボットタイプに応じて設定関数を選択
    if args.robot_type == "go2":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_go2_cfgs()
    elif args.robot_type == "minicheetah":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_minicheetah_cfgs()
    elif args.robot_type == "laikago":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_laikago_cfgs()
    elif args.robot_type == "unitreea1":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_unitreea1_cfgs()
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

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

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
    
    # ファインチューニングの判定とモデル読み込み
    if args.pretrained_path:
        if os.path.exists(args.pretrained_path):
            print(f"Fine-tuning mode: Loading pretrained model from {args.pretrained_path}")
            runner.load(args.pretrained_path)
        else:
            print(f"Warning: Pretrained model not found at {args.pretrained_path}")
            print("Starting training from scratch...")

    # PPO訓練を実行
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
使用例:

# 訓練 + データ収集（デフォルト）
python scripts/train_and_collect.py -e go2-walking -r go2 --max_iterations 301

# 訓練のみ（データ収集なし）
python scripts/train_and_collect.py -e go2-train-only -r go2 --max_iterations 301 --no_collect_data

# データ収集頻度を指定
python scripts/train_and_collect.py -e go2-walking -r go2 --max_iterations 301 --collect_every 20

# ドメインランダマイゼーション + データ収集
python scripts/train_and_collect.py -e go2-dr -r go2 --domain_randomization --mass_range_min 0.8 --mass_range_max 1.2 --max_iterations 301

# ファインチューニング
python scripts/train_and_collect.py -e go2-finetuned -r go2 --pretrained_path logs/go2-walking/model_300.pt --max_iterations 101

# MiniCheetahで訓練
python scripts/train_and_collect.py -e minicheetah-walking -r minicheetah --max_iterations 301 --data_output_dir data/go2_trajectories/minicheetah_data
"""
