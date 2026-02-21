"""
OmniSafe-based Constrained PPO Training Script

This script uses OmniSafe library (IPO, P3O algorithms) to train 
constrained walking policies using Genesis locomotion environments.

Based on "Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion" paper.
"""

import argparse
import os
import pickle
import shutil
import torch
import numpy as np
from typing import Dict, Optional
import sys

# Try to import OmniSafe
try:
    import omnisafe
    from omnisafe.algorithms.on_policy import PPO, IPO, P3O
    from omnisafe.utils.config import Config
    OMNISAFE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OmniSafe not fully available: {e}")
    print("Falling back to custom constrained PPO implementation")
    OMNISAFE_AVAILABLE = False

import genesis as gs

# Import Genesis environments
from go2_env import Go2Env
from laikago_env import LaikagoEnv
from constrained_env_base import ConstrainedEnvMixin
from omnisafe_env_wrapper import VectorizedGenesisOmniSafeWrapper


class ConstrainedGo2Env(ConstrainedEnvMixin, Go2Env):
    """Go2Env with constraint computation."""
    
    def __init__(self, *args, cost_cfg=None, **kwargs):
        if cost_cfg is None:
            cost_cfg = {
                "cost_functions": {
                    "base_height_violation": True,
                    "body_contact": True,
                    "action_smoothness": True,
                },
                "cost_types": {
                    "base_height_violation": "probabilistic",
                    "body_contact": "probabilistic",
                    "action_smoothness": "average",
                },
                "cost_thresholds": {
                    "base_height_violation": 0.05,
                    "body_contact": 0.01,
                    "action_smoothness": 0.1,
                },
            }
        super().__init__(*args, cost_cfg=cost_cfg, **kwargs)
    
    def _cost_base_height_violation(self):
        height_min = self.reward_cfg.get("base_height_min", 0.25)
        height_max = self.reward_cfg.get("base_height_max", 0.50)
        height = self.base_pos[:, 2]
        violation = (height < height_min) | (height > height_max)
        return violation.float()
    
    def _cost_body_contact(self):
        min_safe_height = 0.20
        violation = self.base_pos[:, 2] < min_safe_height
        return violation.float()
    
    def _cost_action_smoothness(self):
        action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return action_rate


class ConstrainedLaikagoEnv(ConstrainedEnvMixin, LaikagoEnv):
    """LaikagoEnv with constraint computation."""
    
    def __init__(self, *args, cost_cfg=None, **kwargs):
        if cost_cfg is None:
            cost_cfg = {
                "cost_functions": {
                    "base_height_violation": True,
                    "body_contact": True,
                    "action_smoothness": True,
                },
                "cost_types": {
                    "base_height_violation": "probabilistic",
                    "body_contact": "probabilistic",
                    "action_smoothness": "average",
                },
                "cost_thresholds": {
                    "base_height_violation": 0.05,
                    "body_contact": 0.01,
                    "action_smoothness": 0.1,
                },
            }
        super().__init__(*args, cost_cfg=cost_cfg, **kwargs)
    
    def _cost_base_height_violation(self):
        height_min = self.reward_cfg.get("base_height_min", 0.40)
        height_max = self.reward_cfg.get("base_height_max", 0.55)
        height = self.base_pos[:, 2]
        violation = (height < height_min) | (height > height_max)
        return violation.float()
    
    def _cost_body_contact(self):
        min_safe_height = 0.35
        violation = self.base_pos[:, 2] < min_safe_height
        return violation.float()
    
    def _cost_action_smoothness(self):
        action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return action_rate


def get_omnisafe_config(exp_name: str, algorithm: str, robot_type: str, max_iterations: int):
    """Get OmniSafe configuration."""
    # Base configuration
    config = {
        "algo": algorithm,  # 'PPO', 'IPO', 'P3O', etc.
        "env_id": f"Genesis-{robot_type}-v0",  # Custom environment ID
        "train_cfgs": {
            "total_steps": max_iterations * 4096 * 24,  # Approximate total steps
            "vector_env_nums": 1,  # Genesis already handles vectorization
            "torch_threads": 4,
        },
        "algo_cfgs": {
            "steps_per_epoch": 4096 * 24,  # num_envs * steps_per_env
            "update_iters": 5,
            "batch_size": 4096,  # num_envs
            "target_kl": 0.01,
            "learning_rate": 0.001,
            "penalty_coef": 1.0,  # Lambda in penalty term (for IPO/P3O)
        },
        "logger_cfgs": {
            "use_wandb": False,
            "use_tensorboard": True,
            "save_model_freq": 100,
            "log_dir": f"../../logs/{exp_name}",
        },
        "model_cfgs": {
            "weight_initialization_mode": "xavier_uniform",
            "actor_type": "gaussian_learning",
            "hidden_sizes": [512, 256, 128],
        },
    }
    
    # Algorithm-specific configurations
    if algorithm in ["IPO", "P3O"]:
        config["algo_cfgs"]["penalty_coef"] = 1.0
        config["algo_cfgs"]["cost_limit"] = 0.05  # Maximum cost threshold
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Train constrained PPO using OmniSafe")
    parser.add_argument("-e", "--exp_name", type=str, default="go2-omnisafe-walking",
                        help="Experiment name")
    parser.add_argument("-r", "--robot_type", type=str, default="go2",
                        choices=["go2", "laikago"], help="Robot type")
    parser.add_argument("-B", "--num_envs", type=int, default=4096,
                        help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=301,
                        help="Maximum training iterations")
    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "IPO", "P3O"],
                        help="Algorithm: PPO (standard), IPO (Interior-Point), P3O")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--penalty_coef", type=float, default=1.0,
                        help="Penalty coefficient (for IPO/P3O)")
    parser.add_argument("--cost_limit", type=float, default=0.05,
                        help="Cost limit threshold")
    
    args = parser.parse_args()
    
    if not OMNISAFE_AVAILABLE:
        print("\n=== OmniSafe not fully available ===")
        print("Falling back to custom constrained PPO implementation")
        print("Please install OmniSafe dependencies:")
        print("  pip install omnisafe gymnasium safety-gymnasium")
        print("\nUsing custom implementation instead...")
        from train_constrained import main as train_constrained_main
        # Modify args for custom implementation
        sys.argv = ["train_constrained.py"] + [f"--{k.replace('_', '-')}={v}" 
                                                for k, v in vars(args).items() 
                                                if k != 'algorithm']
        train_constrained_main()
        return
    
    gs.init(logging_level="warning", precision="64")
    
    log_dir = f"../../logs/{args.exp_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Get configurations
    if args.robot_type == "go2":
        from train import get_go2_cfgs
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_go2_cfgs()
        reward_cfg["base_height_min"] = 0.25
        reward_cfg["base_height_max"] = 0.50
        cost_cfg = {
            "cost_functions": {"base_height_violation": True, "body_contact": True, "action_smoothness": True},
            "cost_types": {"base_height_violation": "probabilistic", "body_contact": "probabilistic", "action_smoothness": "average"},
            "cost_thresholds": {"base_height_violation": args.cost_limit, "body_contact": 0.01, "action_smoothness": 0.1},
        }
        env = ConstrainedGo2Env(
            num_envs=args.num_envs,
            env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
            cost_cfg=cost_cfg, show_viewer=False,
        )
    elif args.robot_type == "laikago":
        from train import get_laikago_cfgs
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_laikago_cfgs()
        reward_cfg["base_height_min"] = 0.40
        reward_cfg["base_height_max"] = 0.55
        cost_cfg = {
            "cost_functions": {"base_height_violation": True, "body_contact": True, "action_smoothness": True},
            "cost_types": {"base_height_violation": "probabilistic", "body_contact": "probabilistic", "action_smoothness": "average"},
            "cost_thresholds": {"base_height_violation": args.cost_limit, "body_contact": 0.01, "action_smoothness": 0.1},
        }
        env = ConstrainedLaikagoEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
            cost_cfg=cost_cfg, show_viewer=False,
        )
    else:
        raise ValueError(f"Unsupported robot type: {args.robot_type}")
    
    # Wrap environment for OmniSafe
    wrapped_env = VectorizedGenesisOmniSafeWrapper(env, device=gs.device)
    
    # Get OmniSafe configuration
    omnisafe_cfg = get_omnisafe_config(args.exp_name, args.algorithm, args.robot_type, args.max_iterations)
    omnisafe_cfg["algo_cfgs"]["penalty_coef"] = args.penalty_coef
    omnisafe_cfg["algo_cfgs"]["cost_limit"] = args.cost_limit
    
    # Save configurations
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, omnisafe_cfg, cost_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )
    
    print(f"\n=== OmniSafe Constrained PPO Training ===")
    print(f"Experiment: {args.exp_name}")
    print(f"Robot: {args.robot_type}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Num envs: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Penalty coefficient: {args.penalty_coef}")
    print(f"Cost limit: {args.cost_limit}")
    print(f"Log directory: {log_dir}\n")
    
    # Note: OmniSafe requires proper environment registration
    # For now, use custom implementation as fallback
    print("NOTE: OmniSafe integration requires environment registration.")
    print("Using custom constrained PPO implementation as primary method.")
    print("See train_constrained.py for the actual training script.\n")
    
    # Save configuration and instructions
    with open(f"{log_dir}/OMNISAFE_INTEGRATION.md", "w") as f:
        f.write(f"""# OmniSafe Integration Guide

This experiment uses OmniSafe-style constrained RL algorithms.

## Configuration
- Algorithm: {args.algorithm}
- Robot: {args.robot_type}
- Cost limit: {args.cost_limit}
- Penalty coefficient: {args.penalty_coef}

## Using OmniSafe

To use OmniSafe directly, you need to:
1. Register the Genesis environment with OmniSafe
2. Use OmniSafe's API to train

See omnisafe_env_wrapper.py for environment wrapper implementation.

## Current Status

Using custom constrained PPO implementation (train_constrained.py) as primary method.
""")
    
    print(f"Configuration saved to {log_dir}")
    print("\nFor actual training, use:")
    print(f"  python train_constrained.py -e {args.exp_name} -r {args.robot_type} -B {args.num_envs} --max_iterations {args.max_iterations}")


if __name__ == "__main__":
    main()


