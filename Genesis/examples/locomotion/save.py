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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago", "unitreea1", "anymalc", "spotmicro"], 
                        default="go2", help="Robot type to train")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"../../logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"../../logs/{args.exp_name}/cfgs.pkl", "rb"))
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

    env.cam.start_recording()
    with torch.no_grad():
        for i in range(1000):  # 1000ステップの動画
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            env.cam.render()

    # モデルのディレクトリに保存（log_dir内に保存）
    filename = os.path.join(log_dir, f"{args.exp_name}_model_{args.ckpt}.mp4")
    
    print(f"Log directory: {os.path.abspath(log_dir)}")
    print(f"Saving video to: {os.path.abspath(filename)}")
    print(f"Directory exists: {os.path.exists(log_dir)}")
    print(f"Directory writable: {os.access(log_dir, os.W_OK)}")
    
    env.cam.stop_recording(save_to_filename=filename, fps=30)
    
    print(f"Video saved successfully: {os.path.exists(filename)}")

if __name__ == "__main__":
    main()

"""
# save
python examples/locomotion/save.py -e go2-walking -r go2 --ckpt 100

#mp4の見方
mpv go2-walking_model_100.mp4
"""
