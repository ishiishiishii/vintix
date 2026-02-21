import argparse
import os
import pickle
from importlib import metadata

import torch

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
from train import (
    get_go2_cfgs,
    get_minicheetah_cfgs,
    get_laikago_cfgs,
    get_unitreea1_cfgs,
    get_anymalc_cfgs,
    get_go1_cfgs,
    get_spotmicro_cfgs,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago", "unitreea1", "anymalc", "go1", "spotmicro"], 
                        default="go2", help="Robot type to train")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"../../logs/{args.exp_name}"
    # Load train_cfg from the experiment config file (needed for model loading)
    _, _, _, _, train_cfg = pickle.load(open(f"../../logs/{args.exp_name}/cfgs.pkl", "rb"))
    
    # Use robot-specific configuration for the evaluation environment
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
    
    reward_cfg["reward_scales"] = {}

    if args.robot_type == "go2":
        env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )
    elif args.robot_type == "laikago":
        env = LaikagoEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )
    elif args.robot_type == "unitreea1":
        env = UnitreeA1Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )
    elif args.robot_type == "anymalc":
        env = ANYmalCEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )
    elif args.robot_type == "go1":
        env = Go1Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )
    elif args.robot_type == "spotmicro":
        env = SpotMicroEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/eval.py -e go2-walking -r go2 --ckpt 300
"""
