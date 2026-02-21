"""
Constrained PPO Training Script
Based on "Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion" paper.

This script trains a walking policy using constrained PPO with:
- Reward: velocity tracking (to maximize)
- Costs: safety constraints (height, contact, etc.)

This implementation extends rsl-rl-lib's OnPolicyRunner to support constraint-based RL.
"""

import argparse
import os
import pickle
import shutil
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
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

# Import existing environments and add constraint functionality
from go2_env import Go2Env
from laikago_env import LaikagoEnv
from anymalc_env import ANYmalCEnv
from constrained_env_base import ConstrainedEnvMixin
from constrained_ppo_runner import ConstrainedOnPolicyRunner


class ConstrainedGo2Env(ConstrainedEnvMixin, Go2Env):
    """
    Go2Env with constraint (cost) computation capabilities.
    """
    
    def __init__(self, *args, cost_cfg=None, **kwargs):
        # Initialize cost configuration with defaults
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
                    "base_height_violation": 0.05,  # Maximum violation rate
                    "body_contact": 0.01,  # Maximum contact rate
                    "action_smoothness": 0.1,  # Maximum average action rate
                },
            }
        
        super().__init__(*args, cost_cfg=cost_cfg, **kwargs)
    
    # Cost functions
    def _cost_base_height_violation(self):
        """Probabilistic constraint: base height out of acceptable range."""
        height_min = self.reward_cfg.get("base_height_min", 0.25)
        height_max = self.reward_cfg.get("base_height_max", 0.50)
        height = self.base_pos[:, 2]
        violation = (height < height_min) | (height > height_max)
        return violation.float()
    
    def _cost_body_contact(self):
        """Probabilistic constraint: body/torso contact with ground (proxy via height)."""
        min_safe_height = 0.20
        violation = self.base_pos[:, 2] < min_safe_height
        return violation.float()
    
    def _cost_action_smoothness(self):
        """Average constraint: action smoothness (rate of change)."""
        action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return action_rate


def get_constrained_train_cfg(exp_name, max_iterations, seed=1):
    """Get training configuration for constrained PPO."""
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",  # Use standard PPO, constraints handled in runner
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
            # Note: cost-related parameters are stored separately and handled by ConstrainedOnPolicyRunner
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
            # Note: cost_critic_hidden_dims is handled separately by ConstrainedOnPolicyRunner
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
        "runner_class_name": "ConstrainedOnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 10,  # Save every 10 iterations to monitor progress
        "empirical_normalization": None,
        "seed": seed,
    }
    return train_cfg_dict


# SimpleConstrainedPPORunner is replaced by ConstrainedOnPolicyRunner


def main():
    parser = argparse.ArgumentParser(description="Train constrained PPO policy")
    parser.add_argument("-e", "--exp_name", type=str, default="go2-constrained-walking",
                        help="Experiment name")
    parser.add_argument("-r", "--robot_type", type=str, default="go2",
                        choices=["go2", "laikago", "anymalc"], help="Robot type")
    parser.add_argument("-B", "--num_envs", type=int, default=4096,
                        help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=301,
                        help="Maximum training iterations")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--penalty_coef", type=float, default=1.0,
                        help="Constraint penalty coefficient (lambda)")
    parser.add_argument("--cost_lr", type=float, default=0.001,
                        help="Learning rate for cost critic")
    
    args = parser.parse_args()
    
    gs.init(logging_level="warning", precision="64")
    
    log_dir = f"../../logs/{args.exp_name}"
    
    # Get configurations
    if args.robot_type == "go2":
        from train import get_go2_cfgs
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_go2_cfgs()
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
        reward_cfg["base_height_min"] = 0.25
        reward_cfg["base_height_max"] = 0.50
    elif args.robot_type == "laikago":
        from train import get_laikago_cfgs
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_laikago_cfgs()
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
        reward_cfg["base_height_min"] = 0.40
        reward_cfg["base_height_max"] = 0.55
    elif args.robot_type == "anymalc":
        from train import get_anymalc_cfgs
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_anymalc_cfgs()
        # ANYmalCの報酬設定をそのまま使用
        # reward_scales: tracking_lin_vel=10.0, tracking_ang_vel=0.2, lin_vel_z=-1.0, base_height=-50.0, action_rate=-0.005, similar_to_default=-0.1
        # base_height_target: 0.5m
        # lin_vel_x_range: [0.9, 0.9]
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
                "base_height_violation": 0.05,  # 最大5%違反率
                "body_contact": 0.01,  # 最大1%接触率
                "action_smoothness": 0.1,  # 最大平均アクション率
            },
        }
        # ANYmalCの高さ範囲設定（base_height_target=0.5mを基準）
        reward_cfg["base_height_min"] = 0.45  # 0.5m - 0.05m
        reward_cfg["base_height_max"] = 0.55  # 0.5m + 0.05m
    else:
        raise ValueError(f"Unsupported robot type: {args.robot_type}")
    
    train_cfg = get_constrained_train_cfg(args.exp_name, args.max_iterations, args.seed)
    # Store cost-related config separately (not in PPO algorithm config)
    cost_algorithm_cfg = {
        "penalty_coef": args.penalty_coef,
        "cost_learning_rate": args.cost_lr,
        "adaptive_threshold": True,
    }
    
    # Create log directory
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configurations
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, cost_cfg, cost_algorithm_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )
    
    # Create constrained environment
    if args.robot_type == "go2":
        env = ConstrainedGo2Env(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            cost_cfg=cost_cfg,
            show_viewer=False,
        )
    elif args.robot_type == "laikago":
        class ConstrainedLaikagoEnv(ConstrainedEnvMixin, LaikagoEnv):
            def __init__(self, *args, cost_cfg=None, **kwargs):
                if cost_cfg is None:
                    cost_cfg = {
                        "cost_functions": {"base_height_violation": True, "body_contact": True, "action_smoothness": True},
                        "cost_types": {"base_height_violation": "probabilistic", "body_contact": "probabilistic", "action_smoothness": "average"},
                        "cost_thresholds": {"base_height_violation": 0.05, "body_contact": 0.01, "action_smoothness": 0.1},
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
        
        env = ConstrainedLaikagoEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            cost_cfg=cost_cfg,
            show_viewer=False,
        )
    elif args.robot_type == "anymalc":
        class ConstrainedANYmalCEnv(ConstrainedEnvMixin, ANYmalCEnv):
            """ANYmalCEnv with constraint computation."""
            
            def __init__(self, *args, cost_cfg=None, **kwargs):
                if cost_cfg is None:
                    cost_cfg = {
                        "cost_functions": {"base_height_violation": True, "body_contact": True, "action_smoothness": True},
                        "cost_types": {"base_height_violation": "probabilistic", "body_contact": "probabilistic", "action_smoothness": "average"},
                        "cost_thresholds": {"base_height_violation": 0.05, "body_contact": 0.01, "action_smoothness": 0.1},
                    }
                super().__init__(*args, cost_cfg=cost_cfg, **kwargs)
            
            def _cost_base_height_violation(self):
                """Probabilistic constraint: ベース高さが範囲外（base_height_target=0.5mを基準）"""
                height_min = self.reward_cfg.get("base_height_min", 0.45)
                height_max = self.reward_cfg.get("base_height_max", 0.55)
                height = self.base_pos[:, 2]
                violation = (height < height_min) | (height > height_max)
                return violation.float()
            
            def _cost_body_contact(self):
                """Probabilistic constraint: 胴体が低すぎる場合（安全マージン0.35m）"""
                min_safe_height = 0.35
                violation = self.base_pos[:, 2] < min_safe_height
                return violation.float()
            
            def _cost_action_smoothness(self):
                """Average constraint: アクションの滑らかさ（action_rateペナルティに対応）"""
                action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
                return action_rate
        
        env = ConstrainedANYmalCEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            cost_cfg=cost_cfg,
            show_viewer=False,
        )
    else:
        raise ValueError(f"Unsupported robot type: {args.robot_type}")
    
    print(f"\n=== Constrained PPO Training Setup ===")
    print(f"Experiment: {args.exp_name}")
    print(f"Robot: {args.robot_type}")
    print(f"Num envs: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Penalty coefficient: {args.penalty_coef}")
    print(f"Cost learning rate: {args.cost_lr}")
    print(f"\nCost constraints:")
    for name, threshold in cost_cfg["cost_thresholds"].items():
        cost_type = cost_cfg["cost_types"].get(name, "unknown")
        print(f"  - {name}: {threshold} ({cost_type})")
    print(f"\nLog directory: {log_dir}\n")
    
    # Create constrained runner
    # Store cost algorithm config in train_cfg for ConstrainedOnPolicyRunner
    train_cfg["cost_algorithm"] = cost_algorithm_cfg
    
    runner = ConstrainedOnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=gs.device,
        cost_cfg=cost_cfg,
    )
    
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    print("\n=== Training Complete ===")
    print(f"Results saved to: {log_dir}")
    print(f"To evaluate, use: python eval.py -r {args.robot_type} -e {args.exp_name}")


if __name__ == "__main__":
    main()

