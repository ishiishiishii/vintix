#!/usr/bin/env python3
"""
軽量確認用: Vintix 評価履歴を HDF5 に保存するスクリプト

本研究の主要コードではなく、専門家データと比較するための補助的ツール。
eval_vintix.py を簡略化し、観測・行動・報酬履歴のみを記録する。
"""

import argparse
import pickle
import sys
from pathlib import Path
from collections import deque
from importlib import metadata

import h5py
import numpy as np
import torch

# Genesis locomotion 環境へのパスを追加
GENESIS_LOCOMOTION_PATH = str(Path(__file__).parents[2] / "Genesis" / "examples" / "locomotion")
sys.path.insert(0, GENESIS_LOCOMOTION_PATH)

# rsl_rl バージョンチェック
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

import genesis as gs
from env import Go2Env
from env import MiniCheetahEnv
from env import LaikagoEnv

# Vintix モジュール
sys.path.insert(0, str(Path(__file__).parent.parent))
from vintix.vintix import Vintix


class VintixHistoryBuffer:
    """Vintix 用の履歴バッファ (環境リセット後も保持)"""

    def __init__(self, max_len: int = 1024) -> None:
        self.max_len = max_len
        self.observations = deque(maxlen=max_len)
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.step_nums = deque(maxlen=max_len)
        self.current_step = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.observations.append(obs.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.step_nums.append(self.current_step)
        self.current_step += 1

    def get_context(self, context_len: int) -> list | None:
        if len(self.observations) == 0:
            return None

        obs_list = list(self.observations)[-context_len:]
        act_list = list(self.actions)[-context_len:]
        rew_list = list(self.rewards)[-context_len:]
        step_list = list(self.step_nums)[-context_len:]

        batch = [{
            "observation": torch.tensor(np.array(obs_list), dtype=torch.float32),
            "prev_action": torch.tensor(np.array(act_list), dtype=torch.float32),
            "prev_reward": torch.tensor(np.array(rew_list), dtype=torch.float32).unsqueeze(1),
            "step_num": torch.tensor(step_list, dtype=torch.int32),
            "task_name": "go2_walking_ad",
        }]
        return batch


def create_env(robot_type: str, exp_name: str):
    """Genesis 環境の生成"""
    genesis_root = Path(__file__).parents[2] / "Genesis"
    log_dir = genesis_root / "logs" / exp_name
    cfgs_path = log_dir / "cfgs.pkl"

    if not cfgs_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfgs_path}")

    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(open(cfgs_path, "rb"))

    env_kwargs = dict(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    if robot_type == "go2":
        env = Go2Env(**env_kwargs)
    elif robot_type == "minicheetah":
        env = MiniCheetahEnv(**env_kwargs)
    elif robot_type == "laikago":
        env = LaikagoEnv(**env_kwargs)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

    return env


def save_hdf5(output_path: Path,
              observations: np.ndarray,
              actions: np.ndarray,
              rewards: np.ndarray,
              step_nums: np.ndarray,
              episode_ids: np.ndarray) -> None:
    """専門家データと揃えた形式で HDF5 保存"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_steps = observations.shape[0]
    group_name = f"0-{num_steps - 1}"

    with h5py.File(output_path, "w") as h5:
        grp = h5.create_group(group_name)
        grp.create_dataset("proprio_observation", data=observations.astype(np.float32))
        grp.create_dataset("action", data=actions.astype(np.float32))
        grp.create_dataset("reward", data=rewards.astype(np.float32))
        grp.create_dataset("step_num", data=step_nums.astype(np.int32))
        grp.create_dataset("episode_id", data=episode_ids.astype(np.int32))
        grp.attrs["source"] = "confirm_eval_history_recorder"


def main():
    parser = argparse.ArgumentParser(description="Vintix 評価履歴を HDF5 に保存 (確認用)")
    parser.add_argument("--exp_name", "-e", type=str, default="go2-walking", help="Genesis experiment name")
    parser.add_argument("--robot_type", "-r", type=str,
                        choices=["go2", "minicheetah", "laikago"], default="go2")
    parser.add_argument("--vintix_path", type=str, required=True, help="Vintix モデルのディレクトリ")
    parser.add_argument("--output_path", type=str, required=True, help="保存先 HDF5 パス")
    parser.add_argument("--context_len", type=int, default=1024, help="Vintix へのコンテキスト長")
    parser.add_argument("--max_steps", type=int, default=1000, help="最大ステップ数")
    parser.add_argument("--reset_threshold", type=int, default=1000,
                        help="この歩数に到達したら環境を強制リセット")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="各エピソードの先頭からこのステップ数をウォームアップとしてマーキングする")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    print("=" * 80)
    print("Confirm Eval History Recorder (auxiliary tool)")
    print("=" * 80)
    print(f"Model path : {args.vintix_path}")
    print(f"Output HDF5: {output_path}")
    print(f"Max steps  : {args.max_steps}")
    print("=" * 80)

    gs.init()
    env = create_env(args.robot_type, args.exp_name)
    print(f"✓ Created {args.robot_type} environment")
    print(f"  Observation dim: {env.num_obs}")
    print(f"  Action dim     : {env.num_actions}")

    # Vintix モデル
    vintix_model = Vintix()
    vintix_model.load_model(args.vintix_path)
    vintix_model = vintix_model.to(gs.device)
    vintix_model.eval()
    print("✓ Vintix model loaded")

    history_buffer = VintixHistoryBuffer(max_len=args.context_len)

    obs, _ = env.reset()
    history_buffer.add(obs[0].cpu().numpy(), np.zeros(env.num_actions, dtype=np.float32), 0.0)

    observations = []
    actions = []
    rewards = []
    step_nums = []
    episode_ids = []
    warmup_flags = []

    step_count = 0
    episode_step = 0
    episode_id = 0

    with torch.no_grad():
        while step_count < args.max_steps:
            context = history_buffer.get_context(args.context_len)

            if context is not None:
                for key in context[0]:
                    if isinstance(context[0][key], torch.Tensor):
                        context[0][key] = context[0][key].to(gs.device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_actions, _ = vintix_model(context)

                if isinstance(pred_actions, list):
                    pred_actions = pred_actions[0]

                if pred_actions.dim() == 3:
                    action = pred_actions[0, -1, :].unsqueeze(0).float()
                elif pred_actions.dim() == 2:
                    action = pred_actions[-1, :].unsqueeze(0).float()
                else:
                    raise ValueError(f"Unexpected pred_actions shape: {pred_actions.shape}")
            else:
                action = torch.zeros(1, env.num_actions, device=gs.device)

            obs, rew, dones, infos = env.step(action)

            obs_np = obs[0].cpu().numpy()
            act_np = action[0].cpu().numpy()
            rew_val = float(rew.cpu().numpy()[0])

            history_buffer.add(obs_np, act_np, rew_val)

            observations.append(obs_np)
            actions.append(act_np)
            rewards.append(rew_val)
            step_nums.append(step_count)
            episode_ids.append(episode_id)
            warmup_flags.append(episode_step < args.warmup_steps if args.warmup_steps > 0 else False)

            step_count += 1
            episode_step += 1

            if step_count % 100 == 0 or step_count == args.max_steps:
                print(f"Progress: {step_count}/{args.max_steps} steps "
                      f"(episode step {episode_step})")

            if dones[0] or episode_step >= args.reset_threshold:
                obs, _ = env.reset()
                history_buffer.add(obs[0].cpu().numpy(), np.zeros(env.num_actions, dtype=np.float32), 0.0)
                episode_id += 1
                episode_step = 0

    observations_np = np.array(observations, dtype=np.float32)
    actions_np = np.array(actions, dtype=np.float32)
    rewards_np = np.array(rewards, dtype=np.float32)
    step_nums_np = np.array(step_nums, dtype=np.int32)
    episode_ids_np = np.array(episode_ids, dtype=np.int32)
    warmup_flags_np = np.array(warmup_flags, dtype=np.bool_)

    save_hdf5(output_path, observations_np, actions_np, rewards_np, step_nums_np, episode_ids_np)
    if args.warmup_steps > 0:
        warmup_path = output_path.with_suffix(".warmup.npy")
        np.save(warmup_path, warmup_flags_np)
        print(f"✓ Warmup mask saved: {warmup_path.name}")
    print(f"✓ Saved HDF5: {output_path.absolute()}")
    print(f"  Stored steps: {len(observations_np)}")


if __name__ == "__main__":
    main()

