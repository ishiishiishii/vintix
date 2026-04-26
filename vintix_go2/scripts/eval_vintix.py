#!/usr/bin/env python3
"""
eval_vintix.py — Vintix チェックポイントの Genesis 並列評価
============================================================

【役割】
  学習済み Vintix（``model.pth``）を Genesis ロコモーション環境に載せ、**常に 10 並列環境 × 各 10
  エピソード**で累積報酬を集計する。結果は各チェックポイント直下の
  ``Result/<表示名>/`` に CSV・集計 CSV・読みやすい PNG（大フォント・無題）および
  ``eval_config.json`` で保存する（A1 のロボ id は常に ``a1``。既定の Vintix タスク名は ``<robot>_walking_ad``、A1 は ``a1_walking_ad``）。
  オプションで評価後に単一環境の短い録画 MP4 も付けられる。

【前提・実行場所】
  リポジトリ内 ``vintix_go2`` をカレントにし、Genesis ロコモーション用の依存（``genesis``、
  ``rsl-rl-lib==2.2.4`` 等）が import できる環境で実行する。スクリプトは ``Genesis/examples/locomotion``
  を ``sys.path`` に追加する。

【基本的な使い方】
  python scripts/eval_vintix.py \\
      --model_path <チェックポイントディレクトリ> [<追加のチェックポイント> ...] \\
      --robot go2 \\
      --robot minicheetah

  複数 ``--model_path`` は順に評価する。ロボットと Vintix タスクを明示したい場合は
  ``--robot_task go2,go2_walking_ad`` のように **ロボid,タスク名**（カンマ区切り）を繰り返し指定。
  ``--robot`` のみのときはタスク名は ``<robot>_walking_ad`` が使われる（A1 は ``a1_walking_ad``）。

【出力レイアウト（各 model_path ごと）】
  <ckpt>/Result/<表示名>/   例: 既定タスクのみ ``go2``、FT タスク ``go2_go1_to_go2``（A1 は ``a1``）
    episode_returns.csv      … 環境別・エピソード別の累積報酬
    episode_mean_std.csv     … エピソード番号別の平均・標準偏差
    episode_cumulative_reward.png
    eval_config.json

【主なコマンドライン引数】
  --model_path PATH [PATH ...]   必須。``model.pth`` を含むディレクトリ（複数可）。
  -e, --exp_name NAME            Genesis の ``logs/<NAME>/cfgs.pkl`` を読む実験名。省略時はロボごとに既定名。
  --robot ROBOT                  ロボ id のみ（go2, go1, minicheetah, laikago, a1 等。旧ロボ id 表記も ``a1`` に正規化）。繰り返し可。
  --robot_task ROBOT,TASK        ``go2,go2_walking_ad`` のように明示。繰り返し可。
                                 ``--robot`` / ``--robot_task`` のいずれか一方以上が必須。
  --context_len N                Vintix コンテキスト長（訓練に合わせる）。既定 2048。
  --reset_threshold N            環境が終了しない場合のエピソード最大ステップ（タイムアウト）。既定 1000。
  --base_seed N                  乱数シードの基準。並列環境 i は ``base_seed + i``。
  --max_total_steps N            全体のシミュレーション安全上限。既定 500000。
  --trajectory_stats_path PATH   メタデータに無いタスク用の軌道統計 JSON。
  --watch_during_eval            評価中に Genesis ビューアを出す。
  --show_viewer                  上記と同義。
  --record_video                 各 (ckpt, robot, task) の評価のあと、単一環境 1000 ステップで
                                 ``recording.mp4`` を同じ Result サブディレクトリに保存。

【重力ドメインランダム化（任意）】
  --domain_gravity_range MIN MAX を付けると有効。**標準重力 |g|=9.81 に対する倍率**の区間
  ``[MIN, MAX]`` で指定する。各並列環境・**各エピソードの開始時**に ``Uniform(MIN, MAX)`` で
  1 点をサンプルし、そのエピソード中の重力に用いる。``MIN == MAX``（例: ``2 2``）のときは
  その固定倍率のみを表す（ランダム幅はゼロ）。有効時はエピソード境界で ``reset_idx`` と
  初期姿勢・関節のランダム化を行う。**質量のドメインランダム化は行わない。**

【例】
  python scripts/eval_vintix.py --model_path runs/exp/epoch_0100 --robot go2
  python scripts/eval_vintix.py --model_path ckpt/a ckpt/b --robot_task go2,ft_go1_to_go2_env
  python scripts/eval_vintix.py --model_path ckpt --robot go2 --domain_gravity_range 0.9 1.1
  python scripts/eval_vintix.py --model_path ckpt --robot go2 --domain_gravity_range 2 2
"""
import argparse
import csv
import os
import subprocess
import pickle
import sys
import json
from pathlib import Path
from collections import deque
from importlib import metadata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np

# Genesis locomotion環境のインポート用
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
from env import A1Env
from env import Go1Env

# Vintixモジュールのインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from vintix.vintix import Vintix

# --- robot id + output naming helpers (keep in-file; do not add new modules) ---
def canon_robot_type(s: str) -> str:
    """Normalize robot id; A1 is always ``a1`` (legacy aliases accepted)."""
    if not s:
        return s
    t = s.strip().lower()
    if t in ("a1", "unitreea1"):
        return "a1"
    return t


_DEFAULT_TASK = {
    "go2": "go2_walking_ad",
    "go1": "go1_walking_ad",
    "a1": "a1_walking_ad",
    "minicheetah": "minicheetah_walking_ad",
    "laikago": "laikago_walking_ad",
}

# Older ``metadata.json`` from checkpoints may still use this task key for A1.
_LEGACY_A1_TASK_IN_METADATA = "unitreea1_walking_ad"


def _checkpoint_task_name_for_model(requested: str, model_metadata: dict) -> str:
    """Align user-facing task name with keys present in loaded checkpoint metadata."""
    if requested in model_metadata:
        return requested
    if (
        requested == "a1_walking_ad"
        and _LEGACY_A1_TASK_IN_METADATA in model_metadata
    ):
        return _LEGACY_A1_TASK_IN_METADATA
    return requested


def default_walking_task_name(robot_type: str) -> str:
    r = canon_robot_type(robot_type)
    if r not in _DEFAULT_TASK:
        raise ValueError(f"No default walking task for robot_type={robot_type!r}")
    return _DEFAULT_TASK[r]


def _display_token(raw: str) -> str:
    t = raw.strip().lower()
    if t in ("unitreea1", "a1"):
        return "a1"
    if t == "go2":
        return "go2"
    if t == "go1":
        return "go1"
    if t == "minicheetah":
        return "minicheetah"
    if t == "laikago":
        return "laikago"
    # keep alnum only
    return "".join(ch for ch in raw.lower() if ch.isalnum())


def eval_result_dir_basename(robot_type: str, task_name: str) -> str:
    """Directory name under ``Result/``. Examples: ``go1``, ``go2_go1_to_go2``."""
    r = canon_robot_type(robot_type)
    disp = r
    if task_name.strip() == default_walking_task_name(r):
        return disp
    tn = task_name.strip()
    if tn.endswith("_env"):
        tn = tn[: -len("_env")]
    if tn.startswith("ft_") and "_to_" in tn:
        body = tn[len("ft_") :]
        src, dst = body.split("_to_", 1)
        return f"{disp}_{_display_token(src)}_to_{_display_token(dst)}"
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in tn).strip("_")
    return f"{disp}_{safe}" if safe else disp


def _write_reward_summary_md(result_root: Path) -> Path:
    """Create Result/reward_summary_by_target_robot.md from per-subdir episode_returns.csv."""
    def _mean_reward_csv(csv_path: Path) -> float:
        vals: list[float] = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return float("nan")
            for row in reader:
                for cell in row[2:]:
                    cell = cell.strip()
                    if not cell:
                        continue
                    vals.append(float(cell))
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    def _fmt_mean_pm_std(xs: list[float]) -> str:
        xs = [v for v in xs if np.isfinite(v)]
        n = len(xs)
        if n == 0:
            return "—"
        m = float(np.mean(xs))
        if n >= 2:
            s = float(np.std(xs, ddof=1))
            return f"{m:.6f} ± {s:.6f}"
        return f"{m:.6f} ± —"

    per_robot: dict[str, list[float]] = {r: [] for r in ("go2", "go1", "a1", "minicheetah")}
    for sub in sorted(result_root.iterdir()):
        if not sub.is_dir():
            continue
        cfg_path = sub / "eval_config.json"
        csv_path = sub / "episode_returns.csv"
        if not (cfg_path.is_file() and csv_path.is_file()):
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            rid = canon_robot_type(str(cfg.get("robot_type", "")).strip())
        except Exception:
            continue
        if rid not in per_robot:
            continue
        per_robot[rid].append(_mean_reward_csv(csv_path))

    lines: list[str] = []
    lines.append("# ターゲットロボット別 累積報酬の要約")
    lines.append("")
    lines.append(
        "各 `Result/<評価結果ディレクトリ>/episode_returns.csv` について、全環境・全エピソードのセルの**算術平均**を1スコアとし、"
        "同一ロボット（列）に属する複数評価結果があれば、その**平均 ± 標準偏差**（サンプル標準偏差、`n=1` のとき SD は `—`）を表示します。"
    )
    lines.append("")
    cols = ("go2", "go1", "a1", "minicheetah")
    lines.append("| 指標 | " + " | ".join(cols) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in cols) + " |")
    lines.append("| 報酬 | " + " | ".join(_fmt_mean_pm_std(per_robot[c]) for c in cols) + " |")
    lines.append("")
    out = result_root / "reward_summary_by_target_robot.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

# 評価は常にこの並列数・エピソード数（CLI では変更不可）
NUM_EVAL_ENVS = 10
NUM_EVAL_EPISODES = 10
# generate_readable_eval_graphs と同系の軸フォント
FONT_SIZE_LABEL = 34
FONT_SIZE_TICK = 28


def _default_trajectory_stats_path(robot_type: str) -> Path:
    """Return default trajectory stats json path for a robot_type, if it exists."""
    # scripts/eval_vintix.py -> vintix_go2/ (parents[1])
    root = Path(__file__).parents[1]
    candidates = {
        "go2": root / "data" / "go2_trajectories" / "go2_trajectories.json",
        "go1": root / "data" / "go1_trajectories" / "go1_trajectories.json",
        "a1": root / "data" / "a1_trajectories" / "a1_trajectories.json",
        "minicheetah": root / "data" / "minicheetah_trajectories" / "minicheetah_trajectories.json",
        "laikago": root / "data" / "laikago_trajectories" / "laikago_trajectories.json",
    }
    return candidates.get(robot_type, Path())


def _load_stats_from_trajectory_json(json_path: Path, task_name: str) -> dict:
    """Load obs/acs normalization stats from a trajectory JSON file."""
    with open(json_path, "r") as f:
        meta = json.load(f)
    stats = {
        task_name: {
            "obs_mean": meta["obs_mean"],
            "obs_std": meta["obs_std"],
            "acs_mean": meta["acs_mean"],
            "acs_std": meta["acs_std"],
        }
    }
    return stats


class VintixHistoryBuffer:
    """Vintix用の履歴バッファ（環境リセット後も保持）"""
    
    def __init__(self, max_len=1024, task_name='go2_walking_ad'):
        self.max_len = max_len
        self.task_name = task_name
        self.observations = deque(maxlen=max_len)
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.step_nums = deque(maxlen=max_len)
        self.current_step = 0
    
    def add(self, obs, action, reward):
        """履歴に追加"""
        self.observations.append(obs.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.step_nums.append(self.current_step)
        self.current_step += 1
    
    def get_context(self, context_len=1024, *, use_bf16: bool = True):
        """Vintix用のコンテキストを取得"""
        if len(self.observations) == 0:
            return None

        obs_list = list(self.observations)[-context_len:]
        act_list = list(self.actions)[-context_len:]
        rew_list = list(self.rewards)[-context_len:]
        step_list = list(self.step_nums)[-context_len:]

        dt = torch.bfloat16 if use_bf16 else torch.float32
        batch = [{
            'observation': torch.tensor(np.array(obs_list), dtype=dt),
            'prev_action': torch.tensor(np.array(act_list), dtype=dt),
            'prev_reward': torch.tensor(np.array(rew_list), dtype=dt).unsqueeze(1),
            'step_num': torch.tensor(step_list, dtype=torch.int32),
            'task_name': self.task_name,
        }]

        return batch
    
    def reset(self):
        """履歴を完全にクリア（通常は使わない）"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.step_nums.clear()
        self.current_step = 0


def _default_exp_name(robot_type: str) -> str:
    if robot_type == "go2":
        return "go2-walking"
    if robot_type == "minicheetah":
        return "mini_cheetah-walking"
    if robot_type == "laikago":
        return "laikago-walking"
    if robot_type == "a1":
        return "a1-walking"
    if robot_type == "go1":
        return "go1-walking"
    raise ValueError(f"Unknown robot_type: {robot_type}")


def _load_env_cfgs(robot_type: str, exp_name: str | None):
    """Load (env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_or_None) from logs or train defaults."""
    genesis_root = Path(__file__).parents[2] / "Genesis"
    log_dir = genesis_root / "logs" / (exp_name or _default_exp_name(robot_type))
    cfgs_path = log_dir / "cfgs.pkl"
    if cfgs_path.exists():
        return pickle.load(open(cfgs_path, "rb"))
    from train import get_go2_cfgs, get_minicheetah_cfgs, get_laikago_cfgs, get_unitreea1_cfgs, get_go1_cfgs

    if robot_type == "go2":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_go2_cfgs()
    elif robot_type == "minicheetah":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_minicheetah_cfgs()
    elif robot_type == "laikago":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_laikago_cfgs()
    elif robot_type == "a1":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_unitreea1_cfgs()
    elif robot_type == "go1":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_go1_cfgs()
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")
    return env_cfg, obs_cfg, reward_cfg, command_cfg, None


def _create_env_from_cfgs(robot_type: str, num_envs: int, cfgs_tuple, show_viewer: bool):
    env_cfg, obs_cfg, reward_cfg, command_cfg, _train = cfgs_tuple[:5]
    if robot_type == "go2":
        return Go2Env(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )
    if robot_type == "minicheetah":
        return MiniCheetahEnv(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )
    if robot_type == "laikago":
        return LaikagoEnv(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )
    if robot_type == "a1":
        return A1Env(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )
    if robot_type == "go1":
        return Go1Env(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )
    raise ValueError(f"Unknown robot type: {robot_type}")


def _ensure_task_on_model(
    vintix_model: Vintix,
    task_name: str,
    robot_type: str,
    trajectory_stats_path: str | None,
) -> None:
    model_metadata = vintix_model.metadata if hasattr(vintix_model, "metadata") else {}
    if task_name in model_metadata:
        return
    stats_path = Path(trajectory_stats_path) if trajectory_stats_path else _default_trajectory_stats_path(robot_type)
    if not stats_path or not stats_path.exists():
        raise ValueError(
            f"Task {task_name!r} not found in model metadata and trajectory stats JSON not found. "
            f"Provide --trajectory_stats_path. Available tasks in model: {list(model_metadata.keys())}"
        )
    stats = _load_stats_from_trajectory_json(stats_path, task_name)
    group_name = "quadruped_locomotion"
    rew_scale = float(stats.get(task_name, {}).get("reward_scale", 1.0)) if isinstance(stats.get(task_name, {}), dict) else 1.0
    vintix_model.add_task(task_name=task_name, group_name=group_name, stats=stats, rew_scale=rew_scale)


def _parse_robot_task_pair(s: str) -> tuple[str, str]:
    parts = s.split(",", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --robot_task {s!r}; use robot_id,task_name "
            "e.g. go2,go2_walking_ad or go2,ft_go1_to_go2_env"
        )
    return canon_robot_type(parts[0]), parts[1].strip()


def _result_subdir(ckpt: Path, robot_type: str, task_name: str) -> Path:
    """``<ckpt>/Result/<表示名>/`` — 例 ``go1``、``go2_go1_to_go2``。"""
    safe = eval_result_dir_basename(robot_type, task_name).replace("/", "_").replace(" ", "_")
    d = ckpt / "Result" / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compute_averaged_stats_from_known_tasks(model_metadata: dict, task_name: str) -> dict:
    """未知タスク用: 既知タスクの obs/acs 統計を要素平均。"""
    obs_means, obs_stds, acs_means, acs_stds = [], [], [], []
    for _tn, meta in model_metadata.items():
        if isinstance(meta, dict) and all(k in meta for k in ("obs_mean", "obs_std", "acs_mean", "acs_std")):
            obs_means.append(np.array(meta["obs_mean"], dtype=np.float64))
            obs_stds.append(np.array(meta["obs_std"], dtype=np.float64))
            acs_means.append(np.array(meta["acs_mean"], dtype=np.float64))
            acs_stds.append(np.array(meta["acs_std"], dtype=np.float64))
    if not obs_means:
        raise ValueError(
            "No task in model_metadata has obs_mean/obs_std/acs_mean/acs_std; cannot average for unknown task."
        )
    return {
        task_name: {
            "obs_mean": np.mean(obs_means, axis=0).tolist(),
            "obs_std": np.mean(obs_stds, axis=0).tolist(),
            "acs_mean": np.mean(acs_means, axis=0).tolist(),
            "acs_std": np.mean(acs_stds, axis=0).tolist(),
        }
    }


def _ensure_task_for_video(
    vintix_model: Vintix,
    task_name: str,
    robot_type: str,
    trajectory_stats_path: str | None,
) -> None:
    """録画用: タスクが無ければ trajectory か平均統計で add_task。"""
    model_metadata = vintix_model.metadata if hasattr(vintix_model, "metadata") else {}
    if task_name in model_metadata:
        return
    _def_traj = _default_trajectory_stats_path(robot_type)
    for cand in filter(
        None,
        [
            Path(trajectory_stats_path) if trajectory_stats_path else None,
            _def_traj if _def_traj.is_file() else None,
        ],
    ):
        if cand.is_file():
            stats = _load_stats_from_trajectory_json(cand, task_name)
            vintix_model.add_task(task_name, "quadruped_locomotion", stats, rew_scale=1.0)
            return
    stats = _compute_averaged_stats_from_known_tasks(model_metadata, task_name)
    vintix_model.add_task(task_name, "quadruped_locomotion", stats, rew_scale=1.0)


RECORDING_STEPS = 1000  # 録画は常にこのステップ数（並列と速度を揃えるため固定）


def _save_readable_episode_cumulative_png(
    mean: np.ndarray, std: np.ndarray, num_episodes: int, out_path: Path
) -> None:
    """題名なし・大フォントの累積報酬曲線を PNG で保存（generate_readable_eval_graphs と同系）。"""
    x = np.arange(1, num_episodes + 1, dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.plot(x, mean, linewidth=2, label="Mean Cumulative Reward per Episode", color="green")
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, color="green", label="±1 Std")
    ax.set_xlabel("Episode Number", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Cumulative Reward per Episode", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)
    ax.set_ylim(-5.0, 28.0)
    ax.set_xlim(0, num_episodes + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_TICK, loc="lower right")
    fig.canvas.draw()
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _record_post_eval_video(
    *,
    ckpt: Path,
    robot_type: str,
    task_name: str,
    exp_name: str | None,
    out_dir: Path,
    trajectory_stats_path: str | None,
    context_len: int,
) -> Path:
    """単一環境で 1000 ステップシミュレーションし MP4 を out_dir に保存。"""
    from genesis.utils.geom import transform_quat_by_quat as transform_quat

    cfgs = _load_env_cfgs(robot_type, exp_name)
    env = _create_env_from_cfgs(robot_type, 1, cfgs, show_viewer=False)
    na = int(env.num_actions)

    print(f"\n{'=' * 80}\nRecording video ({RECORDING_STEPS} steps, single env) robot={robot_type} task={task_name}\n{'=' * 80}")
    vintix_model = Vintix()
    vintix_model.load_model(str(ckpt))
    vintix_model = vintix_model.to(gs.device).to(torch.bfloat16)
    for module in vintix_model.modules():
        if hasattr(module, "alibi_slopes"):
            module.alibi_slopes = module.alibi_slopes.to(torch.bfloat16)
    vintix_model.eval()
    model_meta = vintix_model.metadata if hasattr(vintix_model, "metadata") else {}
    ckpt_task = _checkpoint_task_name_for_model(task_name, model_meta)
    _ensure_task_for_video(vintix_model, ckpt_task, robot_type, trajectory_stats_path)

    history_buffer = VintixHistoryBuffer(max_len=context_len * 2, task_name=ckpt_task)
    obs, _ = env.reset()
    obs = obs[:, :-na]

    env_idx = torch.tensor([0], device=gs.device, dtype=torch.long)
    pos_offset = (torch.rand(1, 3, device=gs.device) - 0.5) * 0.2
    pos_offset[:, 2] = 0.0
    env.base_pos[env_idx] = env.base_init_pos + pos_offset
    env.robot.set_pos(env.base_pos[env_idx], zero_velocity=False, envs_idx=env_idx)

    roll = (torch.rand(1, device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    pitch = (torch.rand(1, device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    quat_noise = torch.stack([cr * cp, cr * sp, sr * cp, -sr * sp], dim=1)
    base_init_quat_expanded = env.base_init_quat.reshape(1, -1).expand(1, -1)
    env.base_quat[env_idx] = transform_quat(base_init_quat_expanded, quat_noise)
    env.robot.set_quat(env.base_quat[env_idx], zero_velocity=False, envs_idx=env_idx)

    dof_noise = (torch.rand(1, env.num_actions, device=gs.device) - 0.5) * 0.2
    env.dof_pos[env_idx] = env.default_dof_pos + dof_noise
    env.robot.set_dofs_position(
        position=env.dof_pos[env_idx],
        dofs_idx_local=env.motors_dof_idx,
        zero_velocity=True,
        envs_idx=env_idx,
    )
    zero_actions = torch.zeros(1, env.num_actions, device=gs.device)
    obs, _, _, _ = env.step(zero_actions)
    obs = obs[:, :-na]

    history_buffer.add(obs[0].cpu().numpy(), np.zeros(env.num_actions), 0.0)

    out_mp4 = out_dir / "recording.mp4"
    fps = 30
    env.cam.start_recording()
    step_count = 0
    with torch.no_grad():
        while step_count < RECORDING_STEPS:
            context = history_buffer.get_context(context_len, use_bf16=False)
            if context is not None:
                for key in context[0]:
                    if isinstance(context[0][key], torch.Tensor):
                        context[0][key] = context[0][key].to(gs.device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_actions, _metadata = vintix_model(context)
                pa = pred_actions[0] if isinstance(pred_actions, list) else pred_actions
                if pa.dim() == 3:
                    action = pa[0, -1, :].unsqueeze(0).float()
                elif pa.dim() == 2:
                    action = pa[-1, :].unsqueeze(0).float()
                else:
                    action = torch.zeros(1, env.num_actions, device=gs.device)
            else:
                action = torch.zeros(1, env.num_actions, device=gs.device)

            obs, rewards, dones, _infos = env.step(action)
            obs = obs[:, :-na]
            env.cam.render()
            history_buffer.add(
                obs[0].cpu().numpy(),
                action[0].cpu().numpy(),
                float(rewards.cpu().numpy()[0]),
            )
            step_count += 1
            if dones[0]:
                reset_indices = torch.tensor([0], device=gs.device, dtype=torch.long)
                env.reset_idx(reset_indices)
                obs[0] = env.obs_buf[0, :-na]

    env.cam.stop_recording(save_to_filename=str(out_mp4), fps=fps)
    print(f"✓ Video saved: {out_mp4}")
    return out_mp4


# --- gravity domain randomization (per env, resampled each episode; no mass DR) ---
def set_gravity_for_envs(env, gravity_scales, env_indices=None, verbose: bool = False) -> None:
    """Per-environment gravity scale vs standard (0, 0, -9.81)."""
    try:
        default_gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        if env_indices is None:
            env_indices = list(range(len(gravity_scales)))
        for i, env_idx in enumerate(env_indices):
            gscale = gravity_scales[i] if isinstance(gravity_scales, (list, np.ndarray)) else gravity_scales
            new_gravity = default_gravity * float(gscale)
            env.scene.sim.set_gravity(
                new_gravity.astype(np.float32),
                envs_idx=torch.tensor([env_idx], dtype=torch.long, device=gs.device),
            )
            if verbose:
                print(f"  Set gravity for env {env_idx}: {gscale:.3f}x (vector {new_gravity})")
    except Exception as e:
        print(f"Warning: Could not set gravity: {e}")


def randomize_initial_state(env, env_indices: list[int]) -> None:
    """Randomize base xy offset, roll/pitch, and dof offsets for listed env indices."""
    if not env_indices:
        return
    from genesis.utils.geom import transform_quat_by_quat as transform_quat

    env_indices_tensor = torch.tensor(env_indices, device=gs.device, dtype=torch.long)
    pos_offset = (torch.rand(len(env_indices), 3, device=gs.device) - 0.5) * 0.2
    pos_offset[:, 2] = 0.0
    env.base_pos[env_indices_tensor] = env.base_init_pos + pos_offset
    env.robot.set_pos(env.base_pos[env_indices_tensor], zero_velocity=False, envs_idx=env_indices_tensor)

    roll = (torch.rand(len(env_indices), device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    pitch = (torch.rand(len(env_indices), device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    quat_noise = torch.stack([cr * cp, cr * sp, sr * cp, -sr * sp], dim=1)
    base_init_quat_expanded = env.base_init_quat.reshape(1, -1).expand(len(env_indices), -1)
    env.base_quat[env_indices_tensor] = transform_quat(base_init_quat_expanded, quat_noise)
    env.robot.set_quat(env.base_quat[env_indices_tensor], zero_velocity=False, envs_idx=env_indices_tensor)

    dof_noise = (torch.rand(len(env_indices), env.num_actions, device=gs.device) - 0.5) * 0.2
    env.dof_pos[env_indices_tensor] = env.default_dof_pos + dof_noise
    env.robot.set_dofs_position(
        position=env.dof_pos[env_indices_tensor],
        dofs_idx_local=env.motors_dof_idx,
        zero_velocity=True,
        envs_idx=env_indices_tensor,
    )


def run_episode_curve_loop(
    *,
    vintix_model: Vintix,
    env,
    robot_type: str,
    task_name: str,
    context_len: int,
    num_envs: int,
    num_episodes: int,
    base_seed: int,
    reset_threshold: int,
    trajectory_stats_path: str | None,
    max_total_steps: int,
    domain_gravity_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Returns shape (num_envs, num_episodes) cumulative reward per episode (NaN if unfinished)."""
    model_meta = vintix_model.metadata if hasattr(vintix_model, "metadata") else {}
    ckpt_task = _checkpoint_task_name_for_model(task_name, model_meta)
    _ensure_task_on_model(vintix_model, ckpt_task, robot_type, trajectory_stats_path)

    torch.manual_seed(int(base_seed))
    np.random.seed(int(base_seed))

    rngs = [torch.Generator(device=gs.device) for _ in range(num_envs)]
    for ei, gen in enumerate(rngs):
        gen.manual_seed(int(base_seed) + ei)
    na = int(env.num_actions)
    buffers = [VintixHistoryBuffer(max_len=context_len * 2, task_name=ckpt_task) for _ in range(num_envs)]

    domain_active = domain_gravity_range is not None
    gravity_scales_np: np.ndarray | None = None

    obs, _ = env.reset()
    obs = obs[:, :-na]

    if domain_active:
        lo_g, hi_g = domain_gravity_range
        gravity_scales_np = np.random.uniform(lo_g, hi_g, size=num_envs)
        set_gravity_for_envs(env, gravity_scales_np.tolist(), list(range(num_envs)))
        randomize_initial_state(env, list(range(num_envs)))
        z0 = torch.zeros(num_envs, env.num_actions, device=gs.device)
        obs, _, _, _ = env.step(z0)
        obs = obs[:, :-na]

    obs = obs.to(device=gs.device, dtype=torch.float32)

    episode_returns = [[] for _ in range(num_envs)]
    ep_acc = np.zeros(num_envs, dtype=np.float64)
    ep_steps = np.zeros(num_envs, dtype=np.int64)
    total_steps = 0
    if domain_active:
        total_steps += 1

    def all_done() -> bool:
        return all(len(episode_returns[i]) >= num_episodes for i in range(num_envs))

    zact_np = np.zeros(na, dtype=np.float32)

    with torch.no_grad():
        while total_steps < max_total_steps and not all_done():
            contexts = []
            idxs = []
            for ei in range(num_envs):
                if len(episode_returns[ei]) >= num_episodes:
                    continue
                ctx = buffers[ei].get_context(context_len=context_len, use_bf16=True)
                if ctx is not None:
                    for k, v in ctx[0].items():
                        if isinstance(v, torch.Tensor):
                            ctx[0][k] = v.to(gs.device)
                    contexts.append(ctx[0])
                    idxs.append(ei)

            action = torch.zeros(num_envs, env.num_actions, device=gs.device, dtype=torch.float32)
            for ei in range(num_envs):
                if len(episode_returns[ei]) >= num_episodes:
                    continue
                action[ei] = (
                    torch.randn(env.num_actions, device=gs.device, dtype=torch.float32, generator=rngs[ei]) * 0.1
                )
            if contexts:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds, _meta = vintix_model(contexts)
                for ei, pred in zip(idxs, preds):
                    if pred.dim() == 3:
                        a = pred[0, -1, :]
                    else:
                        a = pred[-1, :]
                    action[ei] = a.float()

            obs_next, reward, done, _info = env.step(action)
            obs_next = obs_next[:, :-na].to(device=gs.device, dtype=torch.float32)
            rew_np = reward.detach().cpu().numpy().reshape(-1)
            obs_np = obs.detach().cpu().numpy()
            act_np = action.detach().cpu().numpy()

            for ei in range(num_envs):
                if len(episode_returns[ei]) >= num_episodes:
                    continue
                buffers[ei].add(obs_np[ei], act_np[ei], float(rew_np[ei]))
                ep_acc[ei] += float(rew_np[ei])
                ep_steps[ei] += 1

            d = done.detach().cpu().numpy().reshape(-1).astype(bool)
            timeout = ep_steps >= reset_threshold
            term = np.logical_or(d, timeout)

            reset_eis: list[int] = []
            for ei in range(num_envs):
                if len(episode_returns[ei]) >= num_episodes:
                    continue
                if term[ei]:
                    episode_returns[ei].append(float(ep_acc[ei]))
                    ep_acc[ei] = 0.0
                    ep_steps[ei] = 0
                    if domain_active and len(episode_returns[ei]) < num_episodes:
                        reset_eis.append(ei)

            if domain_active and reset_eis:
                ridx = torch.tensor(reset_eis, device=gs.device, dtype=torch.long)
                env.reset_idx(ridx)
                lo_g, hi_g = domain_gravity_range
                for j in reset_eis:
                    gravity_scales_np[j] = np.random.uniform(lo_g, hi_g)
                scales_list = [float(gravity_scales_np[j]) for j in reset_eis]
                set_gravity_for_envs(env, scales_list, reset_eis)
                randomize_initial_state(env, reset_eis)
                zr = torch.zeros(num_envs, env.num_actions, device=gs.device)
                obs_rec, _, _, _ = env.step(zr)
                obs_rec = obs_rec[:, :-na].to(device=gs.device, dtype=torch.float32)
                for j in reset_eis:
                    obs_next[j] = obs_rec[j]
                    buffers[j] = VintixHistoryBuffer(max_len=context_len * 2, task_name=ckpt_task)
                    buffers[j].add(obs_next[j].detach().cpu().numpy(), zact_np, 0.0)
                total_steps += 1

            obs = obs_next
            total_steps += 1

    mat = np.full((num_envs, num_episodes), np.nan, dtype=np.float64)
    for ei, lst in enumerate(episode_returns):
        for j, v in enumerate(lst[:num_episodes]):
            mat[ei, j] = v
    return mat


def collect_robot_tasks(args) -> list[tuple[str, str]]:
    """--robot_task と --robot から (robot_type, task_name) のリストを構築（重複除去）。"""
    pairs: list[tuple[str, str]] = []
    for s in args.robot_task or []:
        pairs.append(_parse_robot_task_pair(s))
    robots = getattr(args, "eval_robots", None) or []
    for r in robots:
        rt = canon_robot_type(r)
        pairs.append((rt, default_walking_task_name(rt)))
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def run_standard_eval(args) -> None:
    """10 並列環境 × 10 エピソード。``<ckpt>/Result/<表示名>/`` に CSV・PNG を出力。"""
    specs = collect_robot_tasks(args)
    if not specs:
        raise ValueError("No robot/tasks: provide --robot_task robot,task and/or --robot (repeatable).")

    paths = getattr(args, "model_path_list", None) or list(args.model_path)

    if getattr(args, "record_video_only", False):
        # Record-only mode: do not run the 10×10 evaluation loop (CSV/graphs assumed to exist already).
        # Keep the same recording settings and output locations as --record_video.
        gs.init()
        if sys.platform != "win32":
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            import importlib

            import OpenGL.platform

            importlib.reload(OpenGL.platform)

        for vp in paths:
            ckpt = Path(vp).resolve()
            if not (ckpt / "model.pth").exists():
                raise FileNotFoundError(f"model.pth not found under {ckpt}")
            for robot_type, task_name in specs:
                out_dir = _result_subdir(ckpt, robot_type, task_name)
                _record_post_eval_video(
                    ckpt=ckpt,
                    robot_type=robot_type,
                    task_name=task_name,
                    exp_name=args.exp_name,
                    out_dir=out_dir,
                    trajectory_stats_path=args.trajectory_stats_path,
                    context_len=args.context_len,
                )
        return

    show_eval = bool(getattr(args, "watch_during_eval", False) or getattr(args, "show_viewer", False))
    # Genesis: only one interactive viewer per process. Fan out one subprocess per (ckpt, robot, task).
    if len(specs) > 1 and show_eval and not os.environ.get("VINTIX_EVAL_FANOUT_CHILD"):
        script = Path(__file__).resolve()
        root = script.parent.parent
        for vp in paths:
            ckpt = str(Path(vp).resolve())
            for robot_type, task_name in specs:
                cmd = [
                    sys.executable,
                    "-u",
                    str(script),
                    "--model_path",
                    ckpt,
                    "--robot_task",
                    f"{robot_type},{task_name}",
                ]
                if args.exp_name:
                    cmd += ["-e", args.exp_name]
                if args.context_len != 2048:
                    cmd += ["--context_len", str(args.context_len)]
                if args.reset_threshold != 1000:
                    cmd += ["--reset_threshold", str(args.reset_threshold)]
                if args.base_seed != 0:
                    cmd += ["--base_seed", str(args.base_seed)]
                if args.max_total_steps != 500_000:
                    cmd += ["--max_total_steps", str(args.max_total_steps)]
                if args.trajectory_stats_path:
                    cmd += ["--trajectory_stats_path", str(args.trajectory_stats_path)]
                if getattr(args, "watch_during_eval", False):
                    cmd.append("--watch_during_eval")
                if getattr(args, "show_viewer", False):
                    cmd.append("--show_viewer")
                if getattr(args, "record_video", False):
                    cmd.append("--record_video")
                if getattr(args, "domain_gravity_range", None) is not None:
                    lo, hi = args.domain_gravity_range
                    cmd += ["--domain_gravity_range", str(lo), str(hi)]
                env_fan = {"VINTIX_EVAL_FANOUT_CHILD": "1"}
                print("Fan-out subprocess:", " ".join(cmd))
                subprocess.run(cmd, check=True, cwd=str(root), env={**os.environ, **env_fan})
        return

    gs.init()

    for vp in paths:
        ckpt = Path(vp).resolve()
        if not (ckpt / "model.pth").exists():
            raise FileNotFoundError(f"model.pth not found under {ckpt}")

        print("=" * 80)
        print(f"Vintix evaluation ({NUM_EVAL_ENVS} envs × {NUM_EVAL_EPISODES} episodes)")
        print("=" * 80)
        print(f"Checkpoint: {ckpt}")
        print("=" * 80 + "\n")

        print("Loading Vintix model...")
        vintix_model = Vintix()
        vintix_model.load_model(str(ckpt))
        vintix_model = vintix_model.to(gs.device).to(torch.bfloat16)
        for module in vintix_model.modules():
            if hasattr(module, "alibi_slopes"):
                module.alibi_slopes = module.alibi_slopes.to(torch.bfloat16)
        vintix_model.eval()
        print("✓ Vintix model loaded\n")

        for robot_type, task_name in specs:
            print(f"=== robot={robot_type} task={task_name} ===")
            out_dir = _result_subdir(ckpt, robot_type, task_name)
            cfgs = _load_env_cfgs(robot_type, args.exp_name)
            env = _create_env_from_cfgs(robot_type, NUM_EVAL_ENVS, cfgs, show_eval)

            mat = run_episode_curve_loop(
                vintix_model=vintix_model,
                env=env,
                robot_type=robot_type,
                task_name=task_name,
                context_len=args.context_len,
                num_envs=NUM_EVAL_ENVS,
                num_episodes=NUM_EVAL_EPISODES,
                base_seed=args.base_seed,
                reset_threshold=args.reset_threshold,
                trajectory_stats_path=args.trajectory_stats_path,
                max_total_steps=args.max_total_steps,
                domain_gravity_range=tuple(args.domain_gravity_range)
                if getattr(args, "domain_gravity_range", None) is not None
                else None,
            )

            mean = np.nanmean(mat, axis=0)
            if NUM_EVAL_ENVS > 1:
                std = np.nanstd(mat, axis=0, ddof=1)
            else:
                std = np.zeros_like(mean)
            std = np.where(np.isfinite(std), std, 0.0)

            csv_path = out_dir / "episode_returns.csv"
            summary_path = out_dir / "episode_mean_std.csv"
            png_path = out_dir / "episode_cumulative_reward.png"

            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                header = ["env_id", "seed_offset"] + [f"episode_{j + 1}" for j in range(NUM_EVAL_EPISODES)]
                w.writerow(header)
                for ei in range(NUM_EVAL_ENVS):
                    row = [ei, args.base_seed + ei] + [
                        f"{mat[ei, j]:.8f}" if np.isfinite(mat[ei, j]) else "" for j in range(NUM_EVAL_EPISODES)
                    ]
                    w.writerow(row)

            with open(summary_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "mean_cumulative_reward", "std_cumulative_reward"])
                for j in range(NUM_EVAL_EPISODES):
                    w.writerow([j + 1, mean[j], std[j] if np.isfinite(std[j]) else 0.0])

            _save_readable_episode_cumulative_png(mean, std, NUM_EVAL_EPISODES, png_path)

            cfg_dump = {
                "model_path": str(ckpt),
                "robot_type": robot_type,
                "task_name": task_name,
                "num_envs": NUM_EVAL_ENVS,
                "num_episodes": NUM_EVAL_EPISODES,
                "context_len": args.context_len,
                "base_seed": args.base_seed,
                "reset_threshold": args.reset_threshold,
                "max_total_steps": args.max_total_steps,
                "exp_name": args.exp_name,
                "trajectory_stats_path": args.trajectory_stats_path,
                "watch_during_eval": getattr(args, "watch_during_eval", False),
                "show_viewer": getattr(args, "show_viewer", False),
                "record_video": getattr(args, "record_video", False),
                "recording_steps": RECORDING_STEPS,
                "domain_gravity_range": list(args.domain_gravity_range)
                if getattr(args, "domain_gravity_range", None) is not None
                else None,
            }
            with open(out_dir / "eval_config.json", "w", encoding="utf-8") as f:
                json.dump(cfg_dump, f, indent=2, default=str)

            print(f"Wrote:\n  {csv_path}\n  {summary_path}\n  {png_path}\n  {out_dir / 'eval_config.json'}\n")

        if getattr(args, "record_video", False):
            # On-screen eval may set PYOPENGL_PLATFORM to native/glx; OffscreenRenderer only supports
            # egl / osmesa / pyglet. Force egl for post-eval MP4 capture on Linux.
            if sys.platform != "win32":
                os.environ["PYOPENGL_PLATFORM"] = "egl"
                import importlib

                import OpenGL.platform

                importlib.reload(OpenGL.platform)
            for robot_type, task_name in specs:
                out_dir = _result_subdir(ckpt, robot_type, task_name)
                _record_post_eval_video(
                    ckpt=ckpt,
                    robot_type=robot_type,
                    task_name=task_name,
                    exp_name=args.exp_name,
                    out_dir=out_dir,
                    trajectory_stats_path=args.trajectory_stats_path,
                    context_len=args.context_len,
                )

        # After finishing all robot/tasks for this checkpoint, write a summary table under Result/.
        try:
            summary_path = _write_reward_summary_md(ckpt / "Result")
            print(f"✓ Wrote summary: {summary_path}")
        except Exception as e:
            print(f"warning: failed to write reward summary md under {ckpt / 'Result'}: {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vintix evaluation: always 10 parallel envs × 10 episodes; CSV + readable PNG under each checkpoint.",
    )
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name for loading Genesis env config (cfgs.pkl). If omitted, chosen from robot when possible.",
    )
    parser.add_argument(
        "--model_path",
        nargs="+",
        required=True,
        help="Checkpoint dir(s) containing model.pth each. Multiple dirs = evaluate each in order.",
    )
    parser.add_argument("--context_len", type=int, default=2048, help="Context length (match training)")
    parser.add_argument(
        "--reset_threshold",
        type=int,
        default=1000,
        help="Episode timeout: end episode after this many steps if not terminated by env",
    )
    parser.add_argument(
        "--show_viewer",
        action="store_true",
        help="Show Genesis viewer during parallel evaluation (same as --watch_during_eval).",
    )
    parser.add_argument(
        "--watch_during_eval",
        action="store_true",
        help="Show Genesis viewer while running the 10-env evaluation.",
    )
    parser.add_argument(
        "--trajectory_stats_path",
        type=str,
        default=None,
        help="Trajectory stats JSON for tasks not in model metadata",
    )
    parser.add_argument(
        "--robot",
        dest="eval_robots",
        action="append",
        default=None,
        help="Robot id only (go2, go1, a1, …); uses default Vintix task <robot>_walking_ad (A1 → a1_walking_ad). Repeatable.",
    )
    parser.add_argument(
        "--robot_task",
        action="append",
        default=None,
        metavar="ROBOT,TASK",
        help="Explicit pair: robot id and Vintix task key, comma-separated, e.g. go2,go2_walking_ad. Repeatable.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="After eval for a checkpoint, record single-env MP4 (1000 steps) per robot/task into the same Result subdir.",
    )
    parser.add_argument(
        "--record_video_only",
        action="store_true",
        help="Record single-env MP4 (1000 steps) per robot/task into the Result subdir WITHOUT running the 10×10 evaluation.",
    )
    parser.add_argument("--base_seed", type=int, default=0, help="Base seed; parallel env i uses base_seed+i")
    parser.add_argument("--max_total_steps", type=int, default=500_000, help="Safety cap on total simulation steps")
    parser.add_argument(
        "--domain_gravity_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Gravity scale range vs |g|=9.81. Each parallel env resamples Uniform(MIN,MAX) at every episode start; "
        "MIN==MAX fixes that scale (e.g. 2 2). Enables episode resets + initial-state randomization. Omit to disable.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.domain_gravity_range is not None:
        lo, hi = float(args.domain_gravity_range[0]), float(args.domain_gravity_range[1])
        if not (lo > 0 and hi > 0 and lo <= hi):
            parser.error("--domain_gravity_range requires positive MIN and MAX with MIN <= MAX (MIN==MAX allowed).")

    if not collect_robot_tasks(args):
        parser.error("Specify at least one --robot and/or --robot_task robot,task (repeatable).")
    args.model_path_list = list(args.model_path)
    run_standard_eval(args)


if __name__ == "__main__":
    main()