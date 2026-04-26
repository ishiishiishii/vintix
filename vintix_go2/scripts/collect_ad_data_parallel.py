#!/usr/bin/env python3
"""
並列環境を使用したAlgorithm Distillationデータ収集

各環境が独立してデータを収集し、各環境1つのファイルに保存
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path
import pickle

import h5py
import numpy as np
import torch
from tqdm import tqdm

GENESIS_LOCOMOTION_PATH = str(Path(__file__).parents[2] / "Genesis" / "examples" / "locomotion")
sys.path.insert(0, GENESIS_LOCOMOTION_PATH)

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
from env import Go1Env
from env import A1Env


def default_expert_ckpt_for_robot(robot: str) -> Path:
    """Return a default expert checkpoint path for a given robot."""
    robot = robot.lower()
    genesis_root = Path(__file__).parents[2] / "Genesis"
    logs = genesis_root / "logs"

    mapping = {
        # Common experts used in this repo (adjust if your logdirs differ).
        "go1": logs / "go1-walking" / "model_300.pt",
        "go2": logs / "go2-walking" / "model_300.pt",
        "minicheetah": logs / "minicheetah-walking" / "model_300.pt",
        "a1": logs / "a1-walking" / "model_300.pt",
    }
    if robot not in mapping:
        raise ValueError(
            f"Unknown robot '{robot}'. "
            f"Supported: {sorted(mapping.keys())}. "
            f"Or pass an explicit model path."
        )
    return mapping[robot]


def _slug_robot(robot: str) -> str:
    return robot.strip().lower()


def format_auto_output_dir(*, target_robot: str, source_robot: str | None, bootstrap_enabled: bool) -> str:
    """Auto-format output_dir for this run (used only when enabled by flag)."""
    t = _slug_robot(target_robot)
    base = f"data/{t}_trajectories"
    if not bootstrap_enabled:
        return f"{base}/{t}_ad_parallel"
    if source_robot is None:
        return f"{base}/{t}_ad_parallel_bootstrap"
    s = _slug_robot(source_robot)
    return f"{base}/{t}_ad_parallel_bootstrap/ft_{s}_to_{t}"


def format_finetune_layout_output_dir(
    layout_parent: str, *, source_robot: str, target_robot: str
) -> str:
    """Cross-run output directory.

    - Default: ``data/<layout_parent>/<source>_to_<target>/`` (e.g. ``data/finetune3M/go1_to_go2``).
    - ``layout_parent == "finerandom"``: ``data/finerandom/<target>_trajectories/`` (flat per target;
      use distinct ``trajectories_env_<source>_to_<target>_*.`` filenames — see ``PerEnvADDataCollector``).
    """
    s = _slug_robot(source_robot)
    t = _slug_robot(target_robot)
    parent = layout_parent.strip().strip("/").replace("..", "")
    if not parent:
        raise ValueError("finetune_layout_parent must be non-empty when set")
    if parent.lower() == "finerandom":
        return f"data/finerandom/{t}_trajectories"
    return f"data/{parent}/{s}_to_{t}"


def cross_collection_outputs_complete(
    output_dir: str | Path,
    *,
    num_envs: int,
    target_steps_per_env: int,
    pair_slug: str | None = None,
) -> bool:
    """Best-effort check whether a prior cross-run finished for all parallel env files."""
    out = Path(output_dir)
    if not out.is_dir():
        return False
    for env_idx in range(int(num_envs)):
        if pair_slug:
            stem = f"trajectories_env_{pair_slug}_{env_idx:04d}"
        else:
            stem = f"trajectories_env_{env_idx:04d}"
        meta_path = out / f"{stem}.json"
        h5_path = out / f"{stem}.h5"
        if not meta_path.is_file() or not h5_path.is_file():
            return False
        if h5_path.stat().st_size <= 0:
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return False
        total_transitions = meta.get("total_transitions")
        if not isinstance(total_transitions, int):
            return False
        if total_transitions < int(target_steps_per_env):
            return False
    return True


def scheduled_epsilon(
    env_total_steps: torch.Tensor,
    threshold_steps: torch.Tensor | None,
    *,
    p: float,
    threshold_valid: bool,
) -> torch.Tensor:
    """Piecewise schedule used for mixing.

    Let ``N_s`` be ``target_steps_per_env``, ``f`` be ``noise_free_fraction``,
    ``T = (1 - f) * N_s``, and ``n_s`` be per-env collected steps ``env_total_steps``.

    Equivalently (with ``r = n_s / ((1 - f) N_s)``), the intended schedule is

        ε(n_s) = ( 1 - ( n_s / ((1 - f) N_s) )^p )^(1/p),

    evaluated with ``r = clamp(n_s / T, 0, 1)`` and ``T = (1-f) N_s`` so ``n_s > T`` gives ``ε = 0``.
    """
    if not threshold_valid or threshold_steps is None:
        return torch.zeros_like(env_total_steps)
    ratios = torch.clamp(env_total_steps / threshold_steps, 0.0, 1.0)
    ratio_term = torch.pow(ratios, p)
    return torch.pow(torch.clamp(1.0 - ratio_term, 0.0), 1.0 / p)


class PerEnvADDataCollector:
    """各環境ごとのADデータ収集器（各環境1つのファイル）"""
    
    def __init__(
        self,
        output_dir,
        env_idx,
        group_size=50000,
        robot_type="go2",
        pair_slug: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_idx = env_idx
        self.group_size = group_size
        self.robot_type = robot_type

        if pair_slug:
            stem = f"trajectories_env_{pair_slug}_{env_idx:04d}"
        else:
            stem = f"trajectories_env_{env_idx:04d}"
        filename = self.output_dir / f"{stem}.h5"
        self.h5_file = h5py.File(filename, 'w')
        self.filename = filename
        
        self.buffer_transitions = []
        self.total_transitions = 0
        self.global_written = 0
        self.group_count = 0
        
        print(f"[Env {env_idx} Collector] Created: {filename}")
    
    def add_transitions_batch(self, transitions):
        """複数のトランジションをまとめて追加
        
        Args:
            transitions: list of transitions, each transition is a dict with keys:
                'obs', 'action', 'next_obs', 'reward', 'step'
        """
        self.buffer_transitions.extend(transitions)
        self.total_transitions += len(transitions)
        
        if len(self.buffer_transitions) >= self.group_size:
            self._flush_buffer()
        
        if self.total_transitions % 10000 == 0:
            self.h5_file.flush()
    
    def _flush_buffer(self):
        """バッファをHDF5に書き込み"""
        if len(self.buffer_transitions) == 0:
            return
        
        chunk_size = min(len(self.buffer_transitions), self.group_size)
        chunk_transitions = self.buffer_transitions[:chunk_size]
        
        obs_list = []
        for i, t in enumerate(chunk_transitions):
            if i == 0:
                obs_list.append(t['obs'])
            obs_list.append(t['next_obs'])
        
        action_list = [t['action'] for t in chunk_transitions]
        reward_list = [t['reward'] for t in chunk_transitions]
        step_num_list = [t['step'] for t in chunk_transitions]
        
        obs_chunk = np.array(obs_list, dtype=np.float32)
        act_chunk = np.array(action_list, dtype=np.float32)
        rew_chunk = np.array(reward_list, dtype=np.float32)
        step_chunk = np.array(step_num_list, dtype=np.int32)
        
        start_idx = self.global_written
        end_idx = start_idx + chunk_size - 1
        group_name = f"{start_idx}-{end_idx}"
        
        group = self.h5_file.create_group(group_name)
        group.create_dataset('proprio_observation', data=obs_chunk)
        group.create_dataset('action', data=act_chunk)
        group.create_dataset('reward', data=rew_chunk)
        group.create_dataset('step_num', data=step_chunk)
        
        del self.buffer_transitions[:chunk_size]
        
        self.global_written += chunk_size
        self.group_count += 1
        
        self.h5_file.flush()
    
    def finalize(self):
        """残りのデータを保存して終了"""
        if len(self.buffer_transitions) > 0:
            self._flush_buffer()
        
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
        print(f"[Env {self.env_idx} Collector] Total: {self.total_transitions:,} transitions")
        self._save_metadata()
    
    def _save_metadata(self):
        """メタデータを保存"""
        if not self.filename.exists():
            return
        
        obs_sum = None
        obs_sq_sum = None
        obs_count = 0
        
        acs_sum = None
        acs_sq_sum = None
        
        reward_sum = 0.0
        reward_sq_sum = 0.0
        reward_min = float('inf')
        reward_max = float('-inf')
        
        with h5py.File(self.filename, 'r') as f:
            for gname in f.keys():
                obs_chunk = np.array(f[gname]['proprio_observation'])
                acs_chunk = np.array(f[gname]['action'])
                rew_chunk = np.array(f[gname]['reward'])
                
                if obs_sum is None:
                    obs_sum = obs_chunk.sum(axis=0)
                    obs_sq_sum = np.square(obs_chunk).sum(axis=0)
                    acs_sum = acs_chunk.sum(axis=0)
                    acs_sq_sum = np.square(acs_chunk).sum(axis=0)
                else:
                    obs_sum += obs_chunk.sum(axis=0)
                    obs_sq_sum += np.square(obs_chunk).sum(axis=0)
                    acs_sum += acs_chunk.sum(axis=0)
                    acs_sq_sum += np.square(acs_chunk).sum(axis=0)
                
                obs_count += obs_chunk.shape[0]
                reward_sum += rew_chunk.sum()
                reward_sq_sum += np.square(rew_chunk).sum()
                reward_min = min(reward_min, float(rew_chunk.min()))
                reward_max = max(reward_max, float(rew_chunk.max()))
        
        if obs_count == 0:
            return
        
        obs_mean = obs_sum / obs_count
        obs_var = np.maximum(obs_sq_sum / obs_count - np.square(obs_mean), 0.0)
        obs_std = np.sqrt(obs_var) + 1e-8
        
        acs_mean = acs_sum / obs_count
        acs_var = np.maximum(acs_sq_sum / obs_count - np.square(acs_mean), 0.0)
        acs_std = np.sqrt(acs_var) + 1e-8
        
        reward_mean = reward_sum / obs_count
        reward_var = max(reward_sq_sum / obs_count - reward_mean ** 2, 0.0)
        reward_std = math.sqrt(reward_var)
        
        # robot_typeに基づいてtask_nameとgroup_nameを設定
        if self.robot_type == "minicheetah":
            task_name = "minicheetah_walking_ad"
            group_name = "minicheetah_locomotion"
        elif self.robot_type == "go1":
            task_name = "go1_walking_ad"
            group_name = "go1_locomotion"
        elif self.robot_type == "a1":
            task_name = "a1_walking_ad"
            group_name = "quadruped_locomotion"
        else:  # go2
            task_name = "go2_walking_ad"
            group_name = "go2_locomotion"
        
        metadata = {
            "task_name": task_name,
            "group_name": group_name,
            "observation_shape": {"proprio": [33]},  # 45次元から行動12次元を除外
            "action_dim": 12,
            "action_type": "continuous",
            "reward_scale": 1.0,
            "algorithm_distillation": True,
            "env_idx": int(self.env_idx),
            "obs_mean": obs_mean.tolist(),
            "obs_std": obs_std.tolist(),
            "acs_mean": acs_mean.tolist(),
            "acs_std": acs_std.tolist(),
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
            "reward_min": float(reward_min),
            "reward_max": float(reward_max),
            "total_transitions": int(self.total_transitions),
        }
        
        metadata_path = self.filename.with_suffix(".json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Collect AD data with parallel environments")
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to trained PPO model (target expert). "
            "Required for single-run mode. Not required when using --run_all_cross."
        ),
    )
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "go1", "a1", "minicheetah"],
                        default="go2", help="Robot type (A1 は ``a1``)")
    default_output_dir = "data/go2_trajectories/go2_ad_parallel"
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                        help="Output directory for AD data")
    parser.add_argument("--target_steps_per_env", type=int, default=1_000_000,
                        help="Total trajectory steps to collect per environment")
    parser.add_argument("--num_envs", type=int, default=10,
                        help="Number of parallel environments (trajectories)")
    parser.add_argument("--noise_free_fraction", type=float, default=0.05,
                        help="Fraction f of trajectory where epsilon=0 (final f*N_s steps are expert-only, default=0.05)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--decay_power", type=float, default=0.6,
                        help="Power p in ε = (1 - r^p)^(1/p) with r = n_s / ((1-f) N_s) (default 0.6)")

    # Optional: replace the "random action" component with a pretrained policy.
    # This is a *non-breaking* extension: if not provided, behavior stays identical.
    parser.add_argument(
        "--bootstrap_model_path",
        type=str,
        default=None,
        help=(
            "Optional. If set, use this PPO policy as the ε-side action instead of uniform random. "
            "Mixture becomes: action = ε * bootstrap_policy + (1-ε) * expert_policy."
        ),
    )
    parser.add_argument(
        "--source_robot",
        type=str,
        default=None,
        help=(
            "Optional shorthand for --bootstrap_model_path. Example: --source_robot go1. "
            "Used only when --bootstrap_model_path is not set."
        ),
    )
    parser.add_argument(
        "--bootstrap_noise_std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise std added to bootstrap-policy actions (default=0.0).",
    )
    parser.add_argument(
        "--finerandom",
        action="store_true",
        help=(
            "Finetuning-style 3-way action mix (requires source expert / bootstrap policy). "
            "Uses the same ε schedule as the usual ε-side: "
            "action = (1-ε)*target_expert + (ε/2)*source_expert + (ε/2)*uniform_random. "
            "Early: half source + half random; late: target expert dominates."
        ),
    )
    parser.add_argument(
        "--auto_output_dir",
        action="store_true",
        help=(
            "If set (and you did not manually override --output_dir), automatically format output_dir. "
            "When bootstrap is enabled and --source_robot is set, it becomes "
            "`data/<target>_trajectories/<target>_ad_parallel_bootstrap/ft_<source>_to_<target>`."
        ),
    )
    parser.add_argument(
        "--finetune_layout_parent",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Optional. If set, bootstrap cross runs save under `data/<NAME>/<source>_to_<target>/` "
            "(e.g. NAME=finetune3M → `data/finetune3M/go1_to_go2`). "
            "NAME=finerandom → `data/finerandom/<target>_trajectories/` with prefixed filenames per pair. "
            "Omit to keep the existing auto output layout (`--auto_output_dir`)."
        ),
    )
    parser.add_argument(
        "--run_all_cross",
        action="store_true",
        help=(
            "Run all cross-robot combinations among {go1, go2, a1, minicheetah} in one command. "
            "Each run collects AD data for the target expert while bootstrapping from the source expert."
        ),
    )
    parser.add_argument(
        "--cross_robots",
        type=str,
        default="go1,go2,a1,minicheetah",
        help="Comma-separated robots used when --run_all_cross is set.",
    )
    parser.add_argument(
        "--run_all_cross_skip_existing",
        action="store_true",
        help=(
            "When using --run_all_cross, skip a (source, target) pair if outputs already look complete "
            "(all trajectories_env_*.json report total_transitions >= --target_steps_per_env). "
            "Requires per-pair output dirs (use --finetune_layout_parent and/or --auto_output_dir)."
        ),
    )
    
    args = parser.parse_args()
    args.robot_type = args.robot_type.strip().lower()
    if args.source_robot is not None:
        args.source_robot = args.source_robot.strip().lower()

    if args.finerandom and not args.run_all_cross:
        if args.bootstrap_model_path is None and args.source_robot is None:
            raise SystemExit("--finerandom requires --source_robot or --bootstrap_model_path")

    if args.run_all_cross:
        robots = [_slug_robot(r) for r in args.cross_robots.split(",") if r.strip()]
        robots = list(dict.fromkeys(robots))
        if len(robots) < 2:
            raise ValueError("--run_all_cross requires at least 2 robots in --cross_robots")

        from subprocess import run as _run

        script_path = Path(__file__).resolve()
        common = [
            sys.executable,
            str(script_path),
            "--target_steps_per_env",
            str(args.target_steps_per_env),
            "--num_envs",
            str(args.num_envs),
            "--noise_free_fraction",
            str(args.noise_free_fraction),
            "--max_steps",
            str(args.max_steps),
            "--decay_power",
            str(args.decay_power),
        ]
        if args.bootstrap_noise_std > 0:
            common += ["--bootstrap_noise_std", str(args.bootstrap_noise_std)]
        if args.finerandom:
            common += ["--finerandom"]
        auto_output_flag = args.finetune_layout_parent is None and args.output_dir == default_output_dir
        if auto_output_flag:
            common += ["--auto_output_dir"]

        print("=" * 80)
        print("RUN ALL CROSS (bootstrap-policy AD collection)")
        print("robots:", robots)
        if args.finetune_layout_parent:
            lp = args.finetune_layout_parent.strip().lower()
            if lp == "finerandom":
                print(
                    "finetune_layout_parent: finerandom -> data/finerandom/<target>_trajectories/ "
                    "(files: trajectories_env_<source>_to_<target>_NNNN.*)"
                )
            else:
                print("finetune_layout_parent:", args.finetune_layout_parent, "-> data/<NAME>/<source>_to_<target>/")
        print("=" * 80)
        for source in robots:
            for target in robots:
                if source == target:
                    continue
                model_path = default_expert_ckpt_for_robot(target)
                cmd = (
                    common
                    + ["--model_path", str(model_path)]
                    + ["-r", target]
                    + ["--source_robot", source]
                )
                if args.finetune_layout_parent:
                    out_dir = format_finetune_layout_output_dir(
                        args.finetune_layout_parent, source_robot=source, target_robot=target
                    )
                    cmd += ["--output_dir", out_dir]
                elif auto_output_flag:
                    out_dir = format_auto_output_dir(
                        target_robot=target,
                        source_robot=source,
                        bootstrap_enabled=True,
                    )
                else:
                    out_dir = None

                if args.run_all_cross_skip_existing:
                    if out_dir is None:
                        raise SystemExit(
                            "--run_all_cross_skip_existing requires per-pair output directories "
                            "(set --finetune_layout_parent, or keep default --output_dir so "
                            "--auto_output_dir is enabled for cross runs)."
                        )
                    pair_slug = f"{_slug_robot(source)}_to_{_slug_robot(target)}"
                    use_slug = (
                        args.finetune_layout_parent is not None
                        and args.finetune_layout_parent.strip().lower() == "finerandom"
                    )
                    if cross_collection_outputs_complete(
                        out_dir,
                        num_envs=int(args.num_envs),
                        target_steps_per_env=int(args.target_steps_per_env),
                        pair_slug=pair_slug if use_slug else None,
                    ):
                        print("\n" + "-" * 80)
                        print(f"SKIP ft_{source}_to_{target} (existing outputs look complete): {out_dir}")
                        print("-" * 80, flush=True)
                        continue
                print("\n" + "-" * 80)
                print(f"COLLECT ft_{source}_to_{target}")
                print("CMD", " ".join(cmd))
                print("-" * 80, flush=True)
                _run(cmd, check=True)
        print("\nALL DONE", flush=True)
        return

    if args.model_path is None:
        raise SystemExit("--model_path is required unless you specify --run_all_cross.")
    
    print("="*80)
    print("Parallel Algorithm Distillation Data Collection")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Robot: {args.robot_type}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Target steps per env: {args.target_steps_per_env}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Decay power (p): {args.decay_power}")

    bootstrap_enabled = (args.bootstrap_model_path is not None) or (args.source_robot is not None)
    if args.output_dir == default_output_dir:
        if args.finetune_layout_parent and args.source_robot is not None:
            args.output_dir = format_finetune_layout_output_dir(
                args.finetune_layout_parent,
                source_robot=args.source_robot,
                target_robot=args.robot_type,
            )
        elif args.auto_output_dir:
            args.output_dir = format_auto_output_dir(
                target_robot=args.robot_type,
                source_robot=args.source_robot,
                bootstrap_enabled=bootstrap_enabled,
            )

    print(f"Output: {args.output_dir}")
    if bootstrap_enabled:
        print("Bootstrap: policy (replacing 'random' component)")
        if args.bootstrap_model_path is not None:
            print(f"  bootstrap_model_path: {args.bootstrap_model_path}")
        else:
            print(f"  source_robot: {args.source_robot} (using default expert ckpt)")
        if args.bootstrap_noise_std > 0:
            print(f"  bootstrap_noise_std: {args.bootstrap_noise_std}")
    else:
        print("Bootstrap: uniform random (default)")
    print("="*80 + "\n")
    
    gs.init(logging_level="warning", precision="64")
    
    model_dir = Path(args.model_path).parent
    cfgs_path = model_dir / "cfgs.pkl"
    
    if not cfgs_path.exists():
        genesis_logs = Path(__file__).parents[2] / "Genesis" / "logs" / model_dir.name
        cfgs_path = genesis_logs / "cfgs.pkl"
        if not cfgs_path.exists():
            raise FileNotFoundError(f"cfgs.pkl not found at {cfgs_path}")
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfgs_path, "rb"))
    
    print("Creating parallel environments...")
    if args.robot_type == "go2":
        env = Go2Env(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    elif args.robot_type == "go1":
        env = Go1Env(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    elif args.robot_type == "a1":
        env = A1Env(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")
    print(f"✓ Created {args.num_envs} parallel {args.robot_type} environments")
    
    # rsl_rl の OnPolicyRunner は train_cfg を破壊的に書き換える（例: policy 内の class_name を pop）。
    # これは「別チェックポイントが別クラス名だから」ではなく、同じ設定でも1回目の初期化で辞書が壊れるため。
    # 対策は「別ファイルの cfg を必須にする」ではなく、Runner ごとに deepcopy して渡すこと。
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), str(model_dir), device=gs.device)
    runner.load(args.model_path)
    policy = runner.get_inference_policy(device=gs.device)
    print(f"✓ Loaded model from {args.model_path}\n")

    bootstrap_policy = None
    if args.bootstrap_model_path is None and args.source_robot is not None:
        args.bootstrap_model_path = str(default_expert_ckpt_for_robot(args.source_robot))
    if args.bootstrap_model_path is not None:
        bootstrap_path = Path(args.bootstrap_model_path)
        if not bootstrap_path.exists():
            raise FileNotFoundError(f"bootstrap_model_path not found: {bootstrap_path}")
        bootstrap_dir = bootstrap_path.parent
        bootstrap_runner = OnPolicyRunner(
            env, copy.deepcopy(train_cfg), str(bootstrap_dir), device=gs.device
        )
        bootstrap_runner.load(str(bootstrap_path))
        bootstrap_policy = bootstrap_runner.get_inference_policy(device=gs.device)
        print(f"✓ Loaded bootstrap policy from {bootstrap_path}\n")

    if args.finerandom and bootstrap_policy is None:
        raise SystemExit("--finerandom requires a loaded bootstrap (source expert) policy")

    # ε側の行動源（ログ・説明用）。実装は下のループで random / bootstrap を切替。
    if args.finerandom:
        mix_side_log = "finerandom_3way"
        mix_side_human = (
            "(1-ε)*target_expert + (ε/2)*source_expert + (ε/2)*uniform; ε=same schedule as ε-side"
        )
    elif bootstrap_policy is None:
        mix_side_log = "uniform_random"
        mix_side_human = "uniform random in normalized action space"
    else:
        mix_side_log = "bootstrap_policy"
        mix_side_human = (
            "bootstrap policy (pretrained PPO; --source_robot or --bootstrap_model_path)"
        )
    
    pair_slug = None
    if (
        args.finetune_layout_parent is not None
        and args.finetune_layout_parent.strip().lower() == "finerandom"
        and args.source_robot is not None
    ):
        pair_slug = f"{_slug_robot(args.source_robot)}_to_{_slug_robot(args.robot_type)}"
    collectors = [
        PerEnvADDataCollector(
            args.output_dir,
            env_idx,
            robot_type=args.robot_type,
            pair_slug=pair_slug,
        )
        for env_idx in range(args.num_envs)
    ]
    
    base_seed = 42
    env_generators = [torch.Generator(device=gs.device) for _ in range(args.num_envs)]
    for env_idx, gen in enumerate(env_generators):
        gen.manual_seed(base_seed + env_idx)
    print(
        f"✓ RNG seeds (base_seed={base_seed}, per-env {base_seed}..{base_seed + args.num_envs - 1}): "
        f"used for ε-side={mix_side_log} "
        f"({'joint-uniform sampling' if bootstrap_policy is None and not args.finerandom else 'optional Gaussian noise on bootstrap actions'})"
    )
    
    default_joint_angles = torch.tensor([
        0.0,   # FR_hip
        0.8,   # FR_thigh
        -1.5,  # FR_calf
        0.0,   # FL_hip
        0.8,   # FL_thigh
        -1.5,  # FL_calf
        0.0,   # RR_hip
        1.0,   # RR_thigh
        -1.5,  # RR_calf
        0.0,   # RL_hip
        1.0,   # RL_thigh
        -1.5,  # RL_calf
    ], device=gs.device)
    
    joint_limits = torch.tensor([
        [-1.0472,  1.0472],   # FR_hip
        [-1.5708,  3.4907],   # FR_thigh
        [-2.7227, -0.83776],  # FR_calf
        [-1.0472,  1.0472],   # FL_hip
        [-1.5708,  3.4907],   # FL_thigh
        [-2.7227, -0.83776],  # FL_calf
        [-1.0472,  1.0472],   # RR_hip
        [-0.5236,  4.5379],   # RR_thigh
        [-2.7227, -0.83776],  # RR_calf
        [-1.0472,  1.0472],   # RL_hip
        [-0.5236,  4.5379],   # RL_thigh
        [-2.7227, -0.83776],  # RL_calf
    ], device=gs.device)
    
    action_scale = env_cfg["action_scale"]
    action_limits = (joint_limits - default_joint_angles.unsqueeze(1)) / action_scale
    
    print(f"✓ Computed action ranges from joint limits (action_scale={action_scale})")
    print(f"  Action ranges per joint:")
    joint_names = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
                   "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
    for i, name in enumerate(joint_names):
        print(f"    {i:2d}: {name:12s} [{action_limits[i, 0]:7.2f}, {action_limits[i, 1]:7.2f}]")
    print()
    
    f = args.noise_free_fraction
    p = args.decay_power
    target_steps_tensor = torch.tensor(float(args.target_steps_per_env), device=gs.device)
    raw_threshold = (1.0 - f) * args.target_steps_per_env
    if raw_threshold <= 0:
        threshold_steps_tensor = None
        threshold_valid = False
    else:
        threshold_steps_tensor = torch.tensor(float(raw_threshold), device=gs.device)
        threshold_valid = True
    
    print(f"Collecting {args.num_envs} independent trajectories with ε schedule...")
    print(f"  Target steps per env: {args.target_steps_per_env:,}")
    print(f"  f (noise-free fraction): {f:.2f}")
    print(f"  p (decay power): {p:.3f}")
    if not threshold_valid:
        print("  ε is 0 from the start (noise-free trajectory)")
    else:
        print(f"  ε becomes 0 for each env after step: {int(threshold_steps_tensor.item()):,}")
    print(f"  ε-side (mixture A): {mix_side_human}")
    print(f"  expert-side (mixture B): target PPO (--model_path)")
    if args.finerandom:
        print(
            "  Method (--finerandom): action = (1-ε)*target_expert + (ε/2)*source_expert + (ε/2)*uniform_random"
        )
    else:
        print(f"  Method: action = ε * ({mix_side_log}) + (1-ε) * expert")
    print(f"  Each environment saves to: trajectories_env_XXXX.h5")
    print()
    
    obs, _ = env.reset()
    # obsは45次元のまま（policyに渡すため）
    
    episode_steps = torch.zeros(args.num_envs, dtype=torch.int32, device=gs.device)
    env_active = torch.ones(args.num_envs, dtype=torch.bool, device=gs.device)
    env_completed = torch.zeros(args.num_envs, dtype=torch.bool, device=gs.device)
    target_reached_flag = torch.zeros(args.num_envs, dtype=torch.bool, device=gs.device)
    completed_env_count = 0
    
    episode_rewards_list = []
    episode_lengths_list = []
    current_episode_rewards = torch.zeros(args.num_envs, device=gs.device)
    env_total_steps = torch.zeros(args.num_envs, dtype=torch.float32, device=gs.device)
    
    env_episode_data = [[] for _ in range(args.num_envs)]
    
    pbar_desc = f"AD collect ({mix_side_log}→expert)" if not args.finerandom else "AD collect (finerandom 3-way)"
    pbar = tqdm(total=args.num_envs, desc=pbar_desc)
    
    with torch.no_grad():
        while bool(env_active.any().item()):
            eps_values = scheduled_epsilon(
                env_total_steps,
                threshold_steps_tensor,
                p=p,
                threshold_valid=threshold_valid,
            )
            eps_values = torch.where(env_active, eps_values, torch.zeros_like(eps_values))

            expert_actions = policy(obs)
            eps_expanded = eps_values.unsqueeze(1)

            if args.finerandom:
                rand_uniform = torch.zeros_like(expert_actions)
                for env_idx in range(args.num_envs):
                    rand_uniform[env_idx] = action_limits[:, 0] + torch.rand(
                        expert_actions[env_idx].shape,
                        device=gs.device,
                        generator=env_generators[env_idx],
                    ) * (action_limits[:, 1] - action_limits[:, 0])
                source_actions = bootstrap_policy(obs)
                if args.bootstrap_noise_std > 0:
                    noise = torch.zeros_like(source_actions)
                    for env_idx in range(args.num_envs):
                        noise[env_idx] = torch.randn(
                            source_actions[env_idx].shape,
                            device=gs.device,
                            generator=env_generators[env_idx],
                        ) * float(args.bootstrap_noise_std)
                    source_actions = source_actions + noise
                g = 1.0 - eps_expanded
                half = eps_expanded * 0.5
                actions = g * expert_actions + half * source_actions + half * rand_uniform
            elif bootstrap_policy is None:
                random_actions = torch.zeros_like(expert_actions)
                for env_idx in range(args.num_envs):
                    random_actions[env_idx] = action_limits[:, 0] + torch.rand(
                        expert_actions[env_idx].shape,
                        device=gs.device,
                        generator=env_generators[env_idx],
                    ) * (action_limits[:, 1] - action_limits[:, 0])
                actions = eps_expanded * random_actions + (1.0 - eps_expanded) * expert_actions
            else:
                # Use a pretrained policy as the ε-side action.
                random_actions = bootstrap_policy(obs)
                if args.bootstrap_noise_std > 0:
                    noise = torch.zeros_like(random_actions)
                    for env_idx in range(args.num_envs):
                        noise[env_idx] = torch.randn(
                            random_actions[env_idx].shape,
                            device=gs.device,
                            generator=env_generators[env_idx],
                        ) * float(args.bootstrap_noise_std)
                    random_actions = random_actions + noise
                actions = eps_expanded * random_actions + (1.0 - eps_expanded) * expert_actions
            actions = torch.clamp(actions, action_limits[:, 0], action_limits[:, 1])
            
            obs_without_actions = obs[:, :-12]
            next_obs, rewards, dones, infos = env.step(actions)
            next_obs_without_actions = next_obs[:, :-12]
            
            for env_idx in range(args.num_envs):
                if env_active[env_idx]:
                    transition = {
                        'obs': obs_without_actions[env_idx].cpu().numpy(),
                        'action': actions[env_idx].cpu().numpy(),
                        'next_obs': next_obs_without_actions[env_idx].cpu().numpy(),
                        'reward': float(rewards[env_idx].cpu()),
                        'step': int(episode_steps[env_idx].cpu())
                    }
                    env_episode_data[env_idx].append(transition)
            
            current_episode_rewards += rewards
            episode_steps += 1
            env_total_steps += env_active.to(env_total_steps.dtype)
            
            done_mask = (dones | (episode_steps >= args.max_steps)) & env_active
            
            if done_mask.any():
                done_indices = torch.where(done_mask)[0]
                
                for idx in done_indices:
                    idx_int = int(idx.cpu())
                    episode_transitions = env_episode_data[idx_int]
                    collectors[idx_int].add_transitions_batch(episode_transitions)
                    
                    ep_reward = float(current_episode_rewards[idx].cpu())
                    ep_length = int(episode_steps[idx].cpu())
                    
                    episode_rewards_list.append(ep_reward)
                    episode_lengths_list.append(ep_length)
                    
                    env_episode_data[idx_int] = []
                    current_episode_rewards[idx] = 0.0
                    episode_steps[idx] = 0
                    
                    if target_reached_flag[idx] or env_total_steps[idx] >= target_steps_tensor:
                        env_active[idx] = False
                        target_reached_flag[idx] = False
                        if not env_completed[idx]:
                            env_completed[idx] = True
                            completed_env_count += 1
                            pbar.update(1)
                            total_steps_so_far = float(env_total_steps.sum().cpu())
                            print(f"\n[Env {completed_env_count}/{args.num_envs} completed] active_envs={int(env_active.sum().cpu())}")
                            print(f"  Collected steps (sum/env): {int(total_steps_so_far):,}")
                
                if len(done_indices) > 0:
                    env.reset_idx(done_indices)
                    # リセット後の観測値からも行動を除外してnext_obs_without_actionsを更新
                    next_obs_without_actions[done_indices] = env.obs_buf[done_indices][:, :-12]
                
                recent_rews = episode_rewards_list[-100:] if len(episode_rewards_list) >= 100 else episode_rewards_list
                recent_lens = episode_lengths_list[-100:] if len(episode_lengths_list) >= 100 else episode_lengths_list
                if env_active.any():
                    active_eps = eps_values[env_active]
                    eps_min = float(active_eps.min().cpu())
                    eps_max = float(active_eps.max().cpu())
                else:
                    eps_min = eps_max = 0.0
                total_steps_so_far = float(env_total_steps.sum().cpu())
                print(f"\n[Progress] ε-range=[{eps_min:.4f}, {eps_max:.4f}], active_envs={int(env_active.sum().cpu())}")
                print(f"  Collected steps (sum/env): {int(total_steps_so_far):,}")
                if recent_rews:
                    print(f"  Last {len(recent_rews)} eps: len={np.mean(recent_lens):.1f}, reward={np.mean(recent_rews):.4f}")
            
            target_reached_mask = (env_total_steps >= target_steps_tensor) & env_active & ~target_reached_flag
            if target_reached_mask.any():
                reached_indices = torch.where(target_reached_mask)[0]
                for idx in reached_indices:
                    target_reached_flag[idx] = True
                    idx_int = int(idx.cpu())
                    print(f"\n[Env {idx_int}] Target steps reached ({int(env_total_steps[idx].cpu()):,} >= {int(target_steps_tensor):,})")
                    print(f"  Waiting for current episode to complete (episode step: {int(episode_steps[idx].cpu())})")
            
            # 次のループで使用する観測値（policyに渡すため45次元のまま）
            obs = next_obs
    
    pbar.close()
    
    for collector in collectors:
        collector.finalize()
    
    print(f"\n{'='*80}")
    print("Parallel AD Data Collection Complete!")
    print(f"{'='*80}")
    print(f"Completed trajectories: {int(env_completed.sum().cpu())} / {args.num_envs}")
    print(f"Total episodes recorded: {len(episode_lengths_list)}")
    total_transitions = sum(c.total_transitions for c in collectors)
    print(f"Total transitions: {total_transitions:,}")
    if episode_lengths_list:
        print(f"\nEpisode Statistics:")
        print(f"  Mean length: {np.mean(episode_lengths_list):.2f}")
        print(f"  Mean reward: {np.mean(episode_rewards_list):.4f}")
        print(f"  Reward std: {np.std(episode_rewards_list):.4f}")
        print(f"  Min reward: {np.min(episode_rewards_list):.4f}")
        print(f"  Max reward: {np.max(episode_rewards_list):.4f}")
    final_eps_values = scheduled_epsilon(
        env_total_steps,
        threshold_steps_tensor,
        p=p,
        threshold_valid=threshold_valid,
    )
    print(f"\nFinal ε range: [{float(final_eps_values.min().cpu()):.4f}, {float(final_eps_values.max().cpu()):.4f}]")
    print(f"Final steps (sum/env): {int(env_total_steps.sum().cpu()):,}")
    print(f"\nOutput files:")
    for collector in collectors:
        print(f"  - {collector.filename}")
        print(f"  - {collector.output_dir / f'trajectories_env_{collector.env_idx:04d}.json'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


"""
使用例（カレントは ``vintix_go2/``、``--decay_power`` は既定 0.6 で省略可）:

# 全ペア（go1, go2, a1, minicheetah）× 10 環境 × 各 30 万ステップ、finerandom 3 成分。
# 学習用の配置は ``data/finerandom/<target>_trajectories/`` にターゲットごとにフラット集約（H5/JSON 名にソースペア接頭辞）。
python scripts/collect_ad_data_parallel.py --run_all_cross \\
    --target_steps_per_env 300000 --num_envs 10 \\
    --finetune_layout_parent finerandom --finerandom

# 単発: ターゲット go2 + ソース go1、自動出力パス:
python scripts/collect_ad_data_parallel.py \\
    --model_path /path/to/Genesis/logs/go2-walking/model_300.pt -r go2 \\
    --source_robot go1 --auto_output_dir

# finerandom レイアウト: ``data/finerandom/<target>_trajectories/trajectories_env_<src>_to_<tgt>_0000.*``
# それ以外: 各環境 ``trajectories_env_0000.h5`` … と対応 json。
"""
