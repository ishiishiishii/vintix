"""
GenLoco-style Training Script for Go2 Robot
Trains generalized locomotion policies using morphological randomization
following GenLoco methodology, but with velocity/height-based rewards instead of
motion capture imitation.

This enables comparison with Vintix/PPO baselines using the same reward structure.
"""

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

from genloco_env import GenLocoGo2Env


def get_train_cfg(exp_name, max_iterations, seed=1):
    """Get training configuration for GenLoco-style training."""
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
        "save_interval": 100,  # Save checkpoint every 100 iterations
        "empirical_normalization": None,
        "seed": seed,
    }
    return train_cfg_dict


def get_genloco_go2_cfgs():
    """
    Get configuration for GenLoco-style Go2 training.
    Uses velocity/height-based rewards (same as Vintix) for fair comparison.
    """
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {  # [rad] - Default standing pose
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
        # PD control parameters (will be randomized in GenLoco)
        "kp": 20.0,
        "kd": 0.5,
        # Termination conditions
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,  # degree
        # Base pose (initial height will be scaled with robot size)
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    
    obs_cfg = {
        "num_obs": 45,  # Same observation space as standard Go2Env
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    
    # Reward configuration - same structure as Vintix/PPO for fair comparison
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,  # Base target (will be scaled with robot size)
        "feet_height_target": 0.075,  # Feet target height
        "reward_scales": {
            "tracking_lin_vel": 1.0,      # Track linear velocity commands
            "tracking_ang_vel": 0.2,      # Track angular velocity commands
            "lin_vel_z": -1.0,            # Penalize vertical velocity
            "base_height": -50.0,         # Penalize deviation from target height
            "action_rate": -0.005,        # Penalize rapid action changes
            "similar_to_default": -0.1,   # Encourage staying near default pose
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],  # Forward velocity command (0.5 m/s)
        "lin_vel_y_range": [0, 0],      # Lateral velocity command
        "ang_vel_range": [0, 0],        # Angular velocity command
    }
    
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser(
        description="GenLoco-style training with morphological randomization"
    )
    parser.add_argument(
        "-e", "--exp_name", type=str, default="genloco-go2",
        help="Experiment name"
    )
    parser.add_argument(
        "-B", "--num_envs", type=int, default=4096,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--max_iterations", type=int, default=301,
        help="Maximum training iterations"
    )
    parser.add_argument(
        "--pretrained_path", type=str, default=None,
        help="Path to pretrained model for fine-tuning"
    )
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Random seed for reproducibility"
    )
    
    # GenLoco morphological randomization parameters
    parser.add_argument(
        "--enable_morphological_randomization", action="store_true", default=True,
        help="Enable GenLoco-style morphological randomization (default: True)"
    )
    parser.add_argument(
        "--size_factor_min", type=float, default=0.8,
        help="Minimum size factor (overall robot scale α) [default: 0.8]"
    )
    parser.add_argument(
        "--size_factor_max", type=float, default=1.2,
        help="Maximum size factor (overall robot scale α) [default: 1.2]"
    )
    parser.add_argument(
        "--mass_range_min", type=float, default=0.7,
        help="Minimum mass multiplier (independent from size) [default: 0.7]"
    )
    parser.add_argument(
        "--mass_range_max", type=float, default=1.3,
        help="Maximum mass multiplier (independent from size) [default: 1.3]"
    )
    parser.add_argument(
        "--kp_range_min", type=float, default=0.9,
        help="Minimum PD position gain multiplier [default: 0.9]"
    )
    parser.add_argument(
        "--kp_range_max", type=float, default=1.1,
        help="Maximum PD position gain multiplier [default: 1.1]"
    )
    parser.add_argument(
        "--kd_range_min", type=float, default=0.9,
        help="Minimum PD velocity gain multiplier [default: 0.9]"
    )
    parser.add_argument(
        "--kd_range_max", type=float, default=1.1,
        help="Maximum PD velocity gain multiplier [default: 1.1]"
    )
    
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init(logging_level="warning", precision="64")
    
    log_dir = f"../../logs/{args.exp_name}"
    
    # Get configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_genloco_go2_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, seed=args.seed)
    
    # GenLoco morphological randomization configuration
    genloco_cfg = {
        "enable_morphological_randomization": args.enable_morphological_randomization,
        "size_factor_range": (args.size_factor_min, args.size_factor_max),
        "mass_range": (args.mass_range_min, args.mass_range_max),
        "kp_range": (args.kp_range_min, args.kp_range_max),
        "kd_range": (args.kd_range_min, args.kd_range_max),
    }
    
    # Check for existing experiment
    resume_existing = (
        args.pretrained_path is None
        and os.path.exists(log_dir)
        and os.path.exists(os.path.join(log_dir, "cfgs.pkl"))
    )
    use_pretrained_in_same_dir = (
        args.pretrained_path is not None
        and os.path.exists(log_dir)
        and os.path.exists(os.path.join(log_dir, "cfgs.pkl"))
    )
    
    # Save or load configurations
    if not resume_existing and not use_pretrained_in_same_dir:
        # New training - create log directory
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, genloco_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )
    elif resume_existing or use_pretrained_in_same_dir:
        # Resume or continue training - load existing configs
        if use_pretrained_in_same_dir:
            print(f"Continuing training in existing log directory: {log_dir}")
        else:
            print(f"Resuming training from existing log directory: {log_dir}")
        configs = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
        if len(configs) == 6:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, genloco_cfg = configs
        else:
            # Legacy format (5 configs)
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = configs
            genloco_cfg = {
                "enable_morphological_randomization": True,
                "size_factor_range": (0.8, 1.2),
                "mass_range": (0.7, 1.3),
                "kp_range": (0.9, 1.1),
                "kd_range": (0.9, 1.1),
            }
        train_cfg["runner"]["max_iterations"] = args.max_iterations
    
    # Create GenLoco environment
    print("\n" + "="*60)
    print("GenLoco-style Morphological Randomization Training")
    print("="*60)
    print(f"Experiment: {args.exp_name}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"\nMorphological Randomization Settings:")
    print(f"  Enabled: {genloco_cfg['enable_morphological_randomization']}")
    if genloco_cfg['enable_morphological_randomization']:
        print(f"  Size factor range: {genloco_cfg['size_factor_range']} (α in GenLoco paper)")
        print(f"  Mass range: {genloco_cfg['mass_range']} (independent from size)")
        print(f"  Kp range: {genloco_cfg['kp_range']}")
        print(f"  Kd range: {genloco_cfg['kd_range']}")
    print(f"\nReward structure: Velocity/Height-based (same as Vintix/PPO)")
    print(f"  - Tracking linear velocity commands")
    print(f"  - Tracking angular velocity commands")
    print(f"  - Maintaining target base height (scaled with robot size)")
    print(f"  - Penalizing vertical velocity and action changes")
    print("="*60 + "\n")
    
    env = GenLocoGo2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        enable_morphological_randomization=genloco_cfg["enable_morphological_randomization"],
        size_factor_range=genloco_cfg["size_factor_range"],
        mass_range=genloco_cfg["mass_range"],
        kp_range=genloco_cfg["kp_range"],
        kd_range=genloco_cfg["kd_range"],
    )
    
    # Create runner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    # Load pretrained model or resume from checkpoint
    if args.pretrained_path:
        if os.path.exists(args.pretrained_path):
            print(f"Fine-tuning mode: Loading pretrained model from {args.pretrained_path}")
            runner.load(args.pretrained_path)
        else:
            print(f"Warning: Pretrained model not found at {args.pretrained_path}")
            print("Starting training from scratch...")
    elif resume_existing:
        # Find latest checkpoint
        import glob
        checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
        if checkpoints:
            latest_checkpoint = max(
                checkpoints,
                key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
            )
            print(f"Resuming training from latest checkpoint: {latest_checkpoint}")
            runner.load(latest_checkpoint)
        else:
            print("No checkpoint found, starting training from scratch...")
    
    # Start training
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    # Print morphological info after training (optional)
    if genloco_cfg["enable_morphological_randomization"]:
        morph_info = env.get_morphological_info()
        if morph_info:
            print("\n" + "="*60)
            print("Final Morphological Randomization Statistics:")
            print("="*60)
            print(f"Size factors - Mean: {morph_info['size_factors'].mean():.3f}, "
                  f"Std: {morph_info['size_factors'].std():.3f}")
            print(f"Mass scales - Mean: {morph_info['mass_scales'].mean():.3f}, "
                  f"Std: {morph_info['mass_scales'].std():.3f}")
            print("="*60 + "\n")


if __name__ == "__main__":
    main()

"""
Example usage:

# Basic GenLoco training with default morphological randomization
python train_genloco.py -e genloco-go2-baseline --num_envs 4096 --max_iterations 301

# GenLoco training with custom morphological ranges (wider variation)
python train_genloco.py -e genloco-go2-wide \
    --size_factor_min 0.7 --size_factor_max 1.3 \
    --mass_range_min 0.6 --mass_range_max 1.4 \
    --num_envs 4096 --max_iterations 301

# GenLoco training with narrower variation (more conservative)
python train_genloco.py -e genloco-go2-narrow \
    --size_factor_min 0.9 --size_factor_max 1.1 \
    --mass_range_min 0.8 --mass_range_max 1.2 \
    --num_envs 4096 --max_iterations 301

# Fine-tuning from a standard PPO model
python train_genloco.py -e genloco-go2-finetune \
    --pretrained_path ../../logs/go2-walking/model_300.pt \
    --num_envs 4096 --max_iterations 101

# Resume training
python train_genloco.py -e genloco-go2-baseline --num_envs 4096 --max_iterations 501

# Disable morphological randomization (baseline comparison)
python train_genloco.py -e genloco-go2-no-morph \
    --enable_morphological_randomization False \
    --num_envs 4096 --max_iterations 301
"""


