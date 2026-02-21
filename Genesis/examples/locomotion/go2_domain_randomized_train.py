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

from go2_domain_randomized_env import Go2DomainRandomizedEnv


def get_train_cfg(exp_name, max_iterations):
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
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
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


def main():
    parser = argparse.ArgumentParser(description="Go2 training with domain randomization")
    parser.add_argument("-e", "--exp_name", type=str, default="go2-domain-randomized")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=101)
    
    # ドメインランダマイゼーション設定
    parser.add_argument("--domain_randomization", action="store_true", default=True,
                       help="Enable domain randomization")
    parser.add_argument("--mass_range", type=float, nargs=2, default=[0.9, 1.1],
                       help="Mass randomization range (min, max)")
    parser.add_argument("--sensor_noise_std", type=float, default=0.005,
                       help="Sensor noise standard deviation")
    parser.add_argument("--enable_sensor_noise", action="store_true", default=True,
                       help="Enable sensor noise")
    
    # 事前学習モデル（オプション）
    parser.add_argument("--pretrained_path", type=str, default=None,
                       help="Path to pretrained model for fine-tuning")
    
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # 既存のログディレクトリを削除
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 設定を保存
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # ドメインランダマイゼーション環境を作成
    env = Go2DomainRandomizedEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        domain_randomization=args.domain_randomization,
        mass_range=tuple(args.mass_range),
        randomize_base_only=True,
        sensor_noise_std=args.sensor_noise_std,
        enable_sensor_noise=args.enable_sensor_noise
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # 事前学習モデルがある場合は読み込み
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained model from: {args.pretrained_path}")
        runner.load(args.pretrained_path)

    print(f"Starting training with domain randomization...")
    print(f"Mass range: {args.mass_range}")
    print(f"Sensor noise std: {args.sensor_noise_std}")
    print(f"Enable sensor noise: {args.enable_sensor_noise}")
    
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    print(f"Training completed! Model saved to: {log_dir}")


if __name__ == "__main__":
    main()

"""
# 使用方法

# 基本的なドメインランダマイゼーション学習
python examples/locomotion/go2_domain_randomized_train.py -e go2-domain-randomized
python examples/locomotion/go2_domain_randomized_train.py --exp_name go2_domain_randomized --max_iterations 101 --mass_range 0.9 1.1 --sensor_noise_std 0.0
# カスタム設定
python examples/locomotion/go2_domain_randomized_train.py -e my-domain-rand --mass_range 0.7 1.3 --friction_range 0.3 1.7 --sensor_noise_std 0.02

# 事前学習モデルからファインチューニング
python examples/locomotion/go2_domain_randomized_train.py -e go2-domain-rand-finetune --pretrained_path logs/go2-walking/model_100.pt
"""
