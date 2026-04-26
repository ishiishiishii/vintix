#!/usr/bin/env python3
"""
Train PPO (Genesis locomotion) while recording the training trajectory for Vintix / AD-style use.

記録形式は ``collect_ad_data_parallel`` / Vintix 用の **Collect 互換 per-env HDF5** のみ
（``trajectories_env_XXXX.h5`` と ``<dirname>.json``）。中間の env_id 混在ファイルは出力しない。

各ファイルのキー: ``proprio_observation``（長さ transitions+1）, ``action``, ``reward``, ``step_num``。
proprio は全観測から末尾 num_actions 次元（直前関節指令）を除いたベクトル。

チェックポイント (model_*.pt, cfgs.pkl, TensorBoard) は Genesis/logs/<exp_name>/。
軌跡のデフォルト保存先: ``vintix_go2/data/ppo_history/``（直下に ``trajectories_env_*.h5`` 等を置く。
``--run_all_cross_ft`` のときのみ上書き防止のため ``data/ppo_history/<source>_to_<target>/`` を明示指定）。

Usage:

    cd /workspace/vintix_go2 && \\
    PYTHONPATH=/workspace/vintix_go2:/workspace/Genesis/examples/locomotion \\
    python scripts/train_with_history.py \\
        -e go2-walking-history \\
        -r go2 \\
        -B 4096 \\
        --max_iterations 301

クロスロボット FT（例: Go1 専門家 → Go2）::

    python scripts/train_with_history.py -r go2 --source_robot go1 --max_iterations 301

→ 履歴は ``data/ppo_history/``（または ``--record_output``）に保存。

12 通りまとめて実行（各 301 イテレーション）::

    python scripts/train_with_history.py --run_all_cross_ft

``--record_output`` に相対パスを渡す場合は **vintix_go2 ディレクトリを基準**に解決される（カレントディレクトリに依存しない）。
Docker で root 実行後にホストユーザーがファイルを扱いやすくするには ``--relax_output_permissions`` または
環境変数 ``VINTIX_RELAX_OUTPUT_PERMISSIONS=1`` を指定する。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_VINTIX_ROOT = _SCRIPT_DIR.parent
_LOCOMOTION_ROOT = _VINTIX_ROOT.parent / "Genesis" / "examples" / "locomotion"
sys.path.insert(0, str(_LOCOMOTION_ROOT))
sys.path.insert(0, str(_VINTIX_ROOT))

def canon_robot_type(s: str) -> str:
    if not s:
        return s
    t = s.strip().lower()
    if t in ("a1", "unitreea1"):
        return "a1"
    return t

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as exc:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from exc

from rsl_rl.runners import OnPolicyRunner

import genesis as gs  # type: ignore

from env import ANYmalCEnv  # type: ignore
from env import Go1Env  # type: ignore
from env import Go2Env  # type: ignore
from env import LaikagoEnv  # type: ignore
from env import MiniCheetahEnv  # type: ignore
from env import A1Env  # type: ignore
from spotmicro_env import SpotMicroEnv  # type: ignore

from train import (  # noqa: E402  (import after sys.path)
    get_anymalc_cfgs,
    get_go1_cfgs,
    get_go2_cfgs,
    get_laikago_cfgs,
    get_minicheetah_cfgs,
    get_spotmicro_cfgs,
    get_train_cfg,
    get_unitreea1_cfgs,
)


def genesis_repo_root() -> Path:
    return _LOCOMOTION_ROOT.parent.parent


def trajectory_dataset_slug(source_robot: Optional[str], robot_type: str) -> str:
    """データセットディレクトリ名: ``<RobotA>_to_<RobotB>``（小文字ロボット id）。"""
    src = source_robot if source_robot is not None else robot_type
    return f"{src}_to_{robot_type}"


# クロスロボット FT: 各ロボットの専門家チェックポイント（Genesis/logs/<run>/model_300.pt）
SOURCE_EXPERT_LOGDIRS = {
    "go1": "go1-walking",
    "go2": "go2-walking",
    "minicheetah": "minicheetah-walking",
    "a1": "a1-walking",
}
DEFAULT_EXPERT_CKPT = "model_300.pt"
CROSS_FT_ROBOTS = ("go1", "go2", "minicheetah", "a1")


def default_pretrained_for_source_robot(source_robot: str) -> Path:
    if source_robot not in SOURCE_EXPERT_LOGDIRS:
        raise ValueError(
            f"source_robot must be one of {list(SOURCE_EXPERT_LOGDIRS.keys())}, got {source_robot!r}"
        )
    return genesis_repo_root() / "logs" / SOURCE_EXPERT_LOGDIRS[source_robot] / DEFAULT_EXPERT_CKPT


def resolve_trajectory_output_dir(record_arg: Optional[str], slug: str) -> Path:
    """履歴保存ディレクトリを決定する。

    - 未指定: ``vintix_go2/data/ppo_history/``（``slug`` は互換のため受け取るが既定では未使用）
    - 相対パス: **vintix_go2 リポジトリルート**からの相対（CWD に依存しない）
    - 絶対パス: そのまま解決
    """
    if record_arg is None:
        return (_VINTIX_ROOT / "data" / "ppo_history").resolve()
    p = Path(record_arg).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_VINTIX_ROOT / p).resolve()


def _chmod_relaxed(path: Path, *, is_dir: bool) -> None:
    """Docker(root) 実行後もホストユーザーが読み書きしやすいよう権限を緩める（任意）。"""
    try:
        path.chmod(0o775 if is_dir else 0o664)
    except OSError:
        pass


# -----------------------------------------------------------------------------
# Per-env HDF5 writer (Collect-compatible, same layout as former convert_ppo_history_to_env_files)
# -----------------------------------------------------------------------------


def _sorted_h5_group_names(f: h5py.File) -> List[str]:
    return sorted(f.keys(), key=lambda x: int(x.split("-")[0]))


@dataclass
class _PerEnvWriter:
    """One env_id → one ``trajectories_env_XXXX.h5`` in Collect layout."""

    output_dir: Path
    env_id: int
    group_size: int = 50000

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"trajectories_env_{self.env_id:04d}.h5"
        self.h5 = h5py.File(self.path, "w")
        self.obs_seq: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.steps: List[int] = []
        self._pending_action: Optional[np.ndarray] = None
        self._pending_reward: Optional[float] = None
        self._pending_step: Optional[int] = None
        self.global_written = 0

    def ingest_row(self, obs: np.ndarray, action: np.ndarray, reward: float, step_num: int) -> None:
        obs = obs.astype(np.float32, copy=False)
        action = action.astype(np.float32, copy=False)
        reward_f = float(reward)
        step_i = int(step_num)

        if not self.obs_seq:
            self.obs_seq.append(obs)
            self._pending_action = action
            self._pending_reward = reward_f
            self._pending_step = step_i
            return

        self.obs_seq.append(obs)
        self.actions.append(self._pending_action)  # type: ignore[arg-type]
        self.rewards.append(self._pending_reward)  # type: ignore[arg-type]
        self.steps.append(self._pending_step)  # type: ignore[arg-type]

        self._pending_action = action
        self._pending_reward = reward_f
        self._pending_step = step_i

        if len(self.actions) >= self.group_size:
            self.flush(finalize_pending=False)

    def flush(self, *, finalize_pending: bool) -> None:
        if finalize_pending and self._pending_action is not None and self.obs_seq:
            self.obs_seq.append(self.obs_seq[-1])
            self.actions.append(self._pending_action)
            self.rewards.append(self._pending_reward if self._pending_reward is not None else 0.0)
            self.steps.append(self._pending_step if self._pending_step is not None else 0)
            self._pending_action = None
            self._pending_reward = None
            self._pending_step = None

        if not self.actions:
            return

        chunk_size = min(len(self.actions), self.group_size)
        obs_chunk = np.stack(self.obs_seq[: chunk_size + 1], axis=0).astype(np.float32, copy=False)
        act_chunk = np.stack(self.actions[:chunk_size], axis=0).astype(np.float32, copy=False)
        rew_chunk = np.asarray(self.rewards[:chunk_size], dtype=np.float32)
        step_chunk = np.asarray(self.steps[:chunk_size], dtype=np.int32)

        start = self.global_written
        end = start + chunk_size - 1
        g = self.h5.create_group(f"{start}-{end}")
        g.create_dataset("proprio_observation", data=obs_chunk)
        g.create_dataset("action", data=act_chunk)
        g.create_dataset("reward", data=rew_chunk)
        g.create_dataset("step_num", data=step_chunk)

        del self.obs_seq[:chunk_size]
        del self.actions[:chunk_size]
        del self.rewards[:chunk_size]
        del self.steps[:chunk_size]
        self.global_written += chunk_size
        self.h5.flush()

    def close(self) -> None:
        self.flush(finalize_pending=True)
        self.h5.close()


def _write_vintix_root_metadata(output_dir: Path) -> None:
    """``<dirname>.json`` for ``MultiTaskMapDataset`` (task_name == directory name)."""
    h5_files = sorted(output_dir.glob("trajectories_env_*.h5"))
    first: Optional[Path] = None
    for p in h5_files:
        with h5py.File(p, "r") as f:
            gnames = _sorted_h5_group_names(f)
            if gnames:
                first = p
                break
    if first is None:
        raise FileNotFoundError(f"No non-empty trajectories_env_*.h5 under {output_dir}")
    task_name = output_dir.name
    with h5py.File(first, "r") as f:
        gnames = _sorted_h5_group_names(f)
        g = f[gnames[0]]
        obs = np.asarray(g["proprio_observation"], dtype=np.float32)
        act = np.asarray(g["action"], dtype=np.float32)
    if obs.ndim < 2:
        raise ValueError(f"Unexpected proprio_observation shape {obs.shape} in {first}")
    obs_dim = int(obs.shape[-1])
    action_dim = int(act.shape[-1]) if act.ndim >= 2 else int(act.size)
    meta = {
        "task_name": task_name,
        "observation_shape": {"proprio": [obs_dim]},
        "action_dim": action_dim,
        "action_type": "continuous",
        "algorithm_distillation": True,
    }
    out_path = output_dir / f"{task_name}.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    print(f"✅ Wrote Vintix root metadata: {out_path}")


# -----------------------------------------------------------------------------
# History recorder (proprio = full obs minus last num_actions, same as Collect)
# -----------------------------------------------------------------------------


class TrainingHistoryRecorder:
    """Stream PPO rollouts directly into Collect-style ``trajectories_env_XXXX.h5`` files."""

    def __init__(
        self,
        output_dir: Path,
        num_envs: int,
        full_obs_dim: int,
        num_actions: int,
        action_dim: int,
        group_size: int = 50000,
        relax_output_permissions: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.relax_output_permissions = relax_output_permissions
        if self.relax_output_permissions:
            _chmod_relaxed(self.output_dir, is_dir=True)
        self.group_size = group_size
        self.full_obs_dim = full_obs_dim
        self.num_actions = num_actions
        self.proprio_dim = full_obs_dim - num_actions
        if self.proprio_dim <= 0:
            raise ValueError(f"Invalid proprio_dim={self.proprio_dim} (full={full_obs_dim}, na={num_actions})")

        self.action_dim = action_dim
        self.num_envs = num_envs

        self.writers: Dict[int, _PerEnvWriter] = {}

        self.step_counters = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_ids = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_reward_running = np.zeros(self.num_envs, dtype=np.float64)

        self.total_transitions_seen = 0
        self.max_iteration = -1

        self.closed = False

    def _get_writer(self, env_id: int) -> _PerEnvWriter:
        if env_id not in self.writers:
            self.writers[env_id] = _PerEnvWriter(
                output_dir=self.output_dir,
                env_id=env_id,
                group_size=self.group_size,
            )
        return self.writers[env_id]

    def record_batch(self, iteration: int, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        """Record one PPO rollout step for all envs (same row order as former interleaved dumps)."""
        if self.closed:
            raise RuntimeError("Recorder is already closed.")

        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        if obs.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} envs, got {obs.shape[0]}")
        if obs.shape[-1] != self.full_obs_dim:
            raise ValueError(f"Expected full obs dim {self.full_obs_dim}, got {obs.shape[-1]}")

        proprio = obs[..., : self.proprio_dim].astype(np.float32)
        step_nums = self.step_counters.copy()

        reward_scalar = rewards.astype(np.float64).flatten()
        self.episode_reward_running += reward_scalar

        for env_i in range(self.num_envs):
            w = self._get_writer(env_i)
            w.ingest_row(
                proprio[env_i],
                actions[env_i].astype(np.float32, copy=False),
                float(rewards[env_i, 0]),
                int(step_nums[env_i]),
            )

        self.total_transitions_seen += self.num_envs
        self.max_iteration = max(self.max_iteration, iteration)

        self.step_counters += 1

        done_indices = np.where(dones.squeeze(-1) > 0.0)[0]
        if done_indices.size > 0:
            for idx in done_indices:
                self.step_counters[idx] = 0
                self.episode_reward_running[idx] = 0.0
                self.episode_ids[idx] += 1

    def finalize(self):
        if self.closed:
            return
        for w in self.writers.values():
            w.close()
        self.closed = True

        _write_vintix_root_metadata(self.output_dir)
        print(
            f"✅ Per-env trajectory HDF5: {self.output_dir} "
            f"({len(self.writers)} env files, {self.total_transitions_seen} transitions logged)"
        )

        if self.relax_output_permissions:
            for fp in self.output_dir.glob("trajectories_env_*.h5"):
                if fp.exists():
                    _chmod_relaxed(fp, is_dir=False)
            meta_json = self.output_dir / f"{self.output_dir.name}.json"
            if meta_json.exists():
                _chmod_relaxed(meta_json, is_dir=False)


# -----------------------------------------------------------------------------
# Custom runner that records history
# -----------------------------------------------------------------------------


class OnPolicyRunnerWithHistory(OnPolicyRunner):
    def __init__(
        self,
        env,
        train_cfg,
        log_dir,
        device,
        recorder: Optional[TrainingHistoryRecorder] = None,
    ):
        super().__init__(env, train_cfg, log_dir, device=device)
        self.history_recorder = recorder

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        recorder = self.history_recorder
        try:
            if self.log_dir is not None and self.writer is None:
                self.logger_type = self.cfg.get("logger", "tensorboard").lower()
                if self.logger_type == "neptune":
                    from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                    self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                    self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
                elif self.logger_type == "wandb":
                    from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                    self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                    self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
                elif self.logger_type == "tensorboard":
                    from torch.utils.tensorboard import SummaryWriter

                    self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
                else:
                    raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

            if init_at_random_ep_len:
                self.env.episode_length_buf = torch.randint_like(
                    self.env.episode_length_buf, high=int(self.env.max_episode_length)
                )

            obs, extras = self.env.get_observations()
            critic_obs = extras["observations"].get("critic", obs)
            obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
            self.train_mode()

            ep_infos = []
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            if self.alg.rnd:
                erewbuffer = deque(maxlen=100)
                irewbuffer = deque(maxlen=100)
                cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
                cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

            start_iter = self.current_learning_iteration
            tot_iter = start_iter + num_learning_iterations
            for it in range(start_iter, tot_iter):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                with torch.inference_mode():
                    for _ in range(self.num_steps_per_env):
                        obs_before_step = obs.detach().to("cpu").numpy() if recorder else None
                        actions = self.alg.act(obs, critic_obs)
                        actions_device = actions.to(self.env.device)
                        actions_cpu = actions.detach().to("cpu").numpy() if recorder else None

                        obs_env, rewards_env, dones_env, infos = self.env.step(actions_device)
                        if recorder:
                            rewards_cpu = rewards_env.detach().to("cpu").numpy()
                            dones_cpu = dones_env.detach().to("cpu").numpy()
                            recorder.record_batch(
                                iteration=it,
                                obs=obs_before_step,
                                actions=actions_cpu,
                                rewards=rewards_cpu,
                                dones=dones_cpu,
                            )

                        obs, rewards, dones = (
                            obs_env.to(self.device),
                            rewards_env.to(self.device),
                            dones_env.to(self.device),
                        )

                        obs = self.obs_normalizer(obs)
                        if "critic" in infos["observations"]:
                            critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"].to(self.device))
                        else:
                            critic_obs = obs

                        self.alg.process_env_step(rewards, dones, infos)
                        intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                        if self.log_dir is not None:
                            if "episode" in infos:
                                ep_infos.append(infos["episode"])
                            elif "log" in infos:
                                ep_infos.append(infos["log"])
                            if self.alg.rnd:
                                cur_ereward_sum += rewards
                                cur_ireward_sum += intrinsic_rewards  # type: ignore
                                cur_reward_sum += rewards + intrinsic_rewards
                            else:
                                cur_reward_sum += rewards
                            cur_episode_length += 1
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0
                            if self.alg.rnd:
                                erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                cur_ereward_sum[new_ids] = 0
                                cur_ireward_sum[new_ids] = 0

                    end.record()
                    torch.cuda.synchronize()
                    collection_time = start.elapsed_time(end) / 1000.0

                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    self.alg.compute_returns(critic_obs)

                mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss = self.alg.update()
                end.record()
                torch.cuda.synchronize()
                learn_time = start.elapsed_time(end) / 1000.0
                self.current_learning_iteration = it

                if self.log_dir is not None:
                    self.log(
                        {
                            "it": it,
                            "num_learning_iterations": num_learning_iterations,
                            "start_iter": start_iter,
                            "tot_iter": tot_iter,
                            "collection_time": collection_time,
                            "learn_time": learn_time,
                            "mean_value_loss": mean_value_loss,
                            "mean_surrogate_loss": mean_surrogate_loss,
                            "mean_entropy": mean_entropy,
                            "mean_rnd_loss": mean_rnd_loss,
                            "mean_symmetry_loss": mean_symmetry_loss,
                            "ep_infos": ep_infos,
                            "rewbuffer": rewbuffer,
                            "lenbuffer": lenbuffer,
                            "erewbuffer": erewbuffer if self.alg.rnd else [],
                            "irewbuffer": irewbuffer if self.alg.rnd else [],
                        }
                    )
                    if it % self.save_interval == 0:
                        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

                ep_infos.clear()

                if it == start_iter:
                    from rsl_rl.utils import store_code_state

                    git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                    if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                        for path in git_file_paths:
                            self.writer.save_file(path)

            if self.log_dir is not None:
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        finally:
            if recorder is not None:
                recorder.finalize()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO and record training trajectory (Collect-compatible proprio).")
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name (Genesis/logs/<exp_name>). Default: go2-walking, or ft_<src>_to_<tgt>_history if --source_robot is set.",
    )
    parser.add_argument(
        "-r",
        "--robot_type",
        type=str,
        choices=["go2", "minicheetah", "laikago", "a1", "anymalc", "go1", "spotmicro"],
        default="go2",
        help="Target robot type to train (fine-tuning destination)",
    )
    parser.add_argument(
        "--source_robot",
        type=str,
        choices=list(SOURCE_EXPERT_LOGDIRS.keys()),
        default=None,
        help="Expert source robot: load Genesis/logs/<expert_run>/model_300.pt as init (overrides need --pretrained_path only if you want a custom file).",
    )
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=301)
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Explicit checkpoint for fine-tuning (if set, overrides --source_robot default)",
    )
    parser.add_argument("--domain_randomization", action="store_true", default=False, help="Enable domain randomization")
    parser.add_argument("--mass_range_min", type=float, default=0.9, help="Minimum mass scale for domain randomization")
    parser.add_argument("--mass_range_max", type=float, default=1.1, help="Maximum mass scale for domain randomization")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument(
        "--record_output",
        type=str,
        default=None,
        help="Directory for per-env HDF5 (default: <vintix_go2>/data/ppo_history). "
        "Relative paths are resolved from the vintix_go2 repository root (not CWD).",
    )
    parser.add_argument("--group_size", type=int, default=50000, help="Transitions per HDF5 flush chunk")
    parser.add_argument(
        "--relax_output_permissions",
        action="store_true",
        help="chmod 775/664 on trajectory dir and files so host user can read/write after Docker (root). "
        "Also enabled if env VINTIX_RELAX_OUTPUT_PERMISSIONS=1.",
    )
    parser.add_argument(
        "--run_all_cross_ft",
        action="store_true",
        help="Run all 12 cross-robot fine-tunings (go1,go2,minicheetah,a1) × 301 iterations each, sequential subprocesses.",
    )
    return parser


def run_all_cross_ft_matrix(args: argparse.Namespace) -> None:
    """4×3 通り: 同一ロボット以外のソース→ターゲット。"""
    script = Path(__file__).resolve()
    env = os.environ.copy()
    extra = f"{_VINTIX_ROOT}:{_LOCOMOTION_ROOT}"
    env["PYTHONPATH"] = f"{extra}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else extra

    pairs = [(s, t) for s in CROSS_FT_ROBOTS for t in CROSS_FT_ROBOTS if s != t]
    print(f"Running {len(pairs)} cross-robot fine-tuning jobs (301 iterations each)...\n")

    for src, tgt in pairs:
        exp = f"ft_{src}_to_{tgt}_history"
        rec = f"data/ppo_history/{src}_to_{tgt}"
        cmd = [
            sys.executable,
            str(script),
            "-r",
            tgt,
            "--source_robot",
            src,
            "-e",
            exp,
            "--record_output",
            rec,
            "--max_iterations",
            "301",
            "-B",
            str(args.num_envs),
            "--group_size",
            str(args.group_size),
            "--seed",
            str(args.seed),
        ]
        if args.relax_output_permissions:
            cmd.append("--relax_output_permissions")
        if args.domain_randomization:
            cmd.append("--domain_randomization")
            cmd += ["--mass_range_min", str(args.mass_range_min), "--mass_range_max", str(args.mass_range_max)]

        print("=" * 80)
        print(" ", " ".join(cmd))
        print("=" * 80)
        subprocess.run(cmd, check=True, cwd=str(_VINTIX_ROOT), env=env)

    print(f"\nCompleted {len(pairs)} jobs.")


def run_training_job(args: argparse.Namespace) -> None:
    if args.source_robot is not None and args.source_robot == args.robot_type:
        raise ValueError("--source_robot と -r（学習先ロボット）は別のロボットにしてください。")

    exp_name = args.exp_name
    if exp_name is None:
        exp_name = f"ft_{args.source_robot}_to_{args.robot_type}_history" if args.source_robot else "go2-walking"

    pretrained_path = args.pretrained_path
    if pretrained_path is None and args.source_robot is not None:
        pp = default_pretrained_for_source_robot(args.source_robot)
        if not pp.is_file():
            raise FileNotFoundError(f"Source expert checkpoint not found: {pp}")
        pretrained_path = str(pp)
        print(f"Using default expert for --source_robot {args.source_robot}: {pretrained_path}")

    slug = trajectory_dataset_slug(args.source_robot, args.robot_type)
    record_arg = args.record_output
    record_output = resolve_trajectory_output_dir(record_arg, slug)
    relax_perm = args.relax_output_permissions or os.environ.get("VINTIX_RELAX_OUTPUT_PERMISSIONS", "").strip() in (
        "1",
        "true",
        "yes",
    )

    gs.init(logging_level="warning", precision="64")

    genesis_root = genesis_repo_root()
    log_dir = os.path.join(str(genesis_root), "logs", exp_name)

    if args.robot_type == "go2":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_go2_cfgs()
    elif args.robot_type == "minicheetah":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_minicheetah_cfgs()
    elif args.robot_type == "laikago":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_laikago_cfgs()
    elif args.robot_type == "a1":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_unitreea1_cfgs()
    elif args.robot_type == "anymalc":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_anymalc_cfgs()
    elif args.robot_type == "go1":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_go1_cfgs()
    elif args.robot_type == "spotmicro":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_spotmicro_cfgs()
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")

    dr_cfg = {
        "domain_randomization": args.domain_randomization,
        "mass_range": (args.mass_range_min, args.mass_range_max),
    }
    train_cfg = get_train_cfg(exp_name, args.max_iterations, seed=args.seed)

    resume_existing = (
        pretrained_path is None and os.path.exists(log_dir) and os.path.exists(os.path.join(log_dir, "cfgs.pkl"))
    )
    use_pretrained_in_same_dir = (
        pretrained_path is not None and os.path.exists(log_dir) and os.path.exists(os.path.join(log_dir, "cfgs.pkl"))
    )

    if not resume_existing and not use_pretrained_in_same_dir:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            open(os.path.join(log_dir, "cfgs.pkl"), "wb"),
        )
        if relax_perm:
            try:
                Path(log_dir).chmod(0o775)
                Path(os.path.join(log_dir, "cfgs.pkl")).chmod(0o664)
            except OSError:
                pass
    elif resume_existing or use_pretrained_in_same_dir:
        if use_pretrained_in_same_dir:
            print(f"Continuing training in existing log directory: {log_dir}")
        else:
            print(f"Resuming training from existing log directory: {log_dir}")
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(os.path.join(log_dir, "cfgs.pkl"), "rb"))
        train_cfg["runner"]["max_iterations"] = args.max_iterations

    full_obs_dim = obs_cfg["num_obs"]
    num_actions = env_cfg["num_actions"]

    if args.robot_type == "go2":
        env = Go2Env(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"],
        )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"],
        )
    elif args.robot_type == "laikago":
        env = LaikagoEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"],
        )
    elif args.robot_type == "a1":
        env = A1Env(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"],
        )
    elif args.robot_type == "anymalc":
        env = ANYmalCEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"],
        )
    elif args.robot_type == "go1":
        env = Go1Env(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"],
        )
    elif args.robot_type == "spotmicro":
        env = SpotMicroEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"],
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")

    if dr_cfg["domain_randomization"]:
        print(f"ドメインランダマイゼーション有効: 質量 {dr_cfg['mass_range'][0]:.2f} - {dr_cfg['mass_range'][1]:.2f}")
    else:
        print("ドメインランダマイゼーション無効")

    recorder = TrainingHistoryRecorder(
        output_dir=record_output,
        num_envs=env.num_envs,
        full_obs_dim=full_obs_dim,
        num_actions=num_actions,
        action_dim=env_cfg["num_actions"],
        group_size=args.group_size,
        relax_output_permissions=relax_perm,
    )

    print(f"PPO checkpoints & TensorBoard: {log_dir}")
    print(f"Trajectory data (Collect-style per-env HDF5): {record_output.resolve()}")

    runner = OnPolicyRunnerWithHistory(env, train_cfg, log_dir, device=gs.device, recorder=recorder)

    if pretrained_path:
        if os.path.exists(pretrained_path):
            print(f"Fine-tuning: loading {pretrained_path}")
            runner.load(pretrained_path)
        else:
            print(f"Warning: pretrained model not found at {pretrained_path}")
    elif resume_existing:
        checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            runner.load(latest_checkpoint)
        else:
            print("No checkpoint found, starting from scratch...")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    if relax_perm:
        ld = Path(log_dir)
        try:
            ld.chmod(0o775)
        except OSError:
            pass
        for pattern in ("model_*.pt", "events.*", "cfgs.pkl"):
            for fp in ld.glob(pattern):
                _chmod_relaxed(fp, is_dir=False)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.robot_type = canon_robot_type(args.robot_type)
    if args.source_robot is not None:
        args.source_robot = canon_robot_type(args.source_robot)
    if args.run_all_cross_ft:
        run_all_cross_ft_matrix(args)
        return
    run_training_job(args)


if __name__ == "__main__":
    main()
