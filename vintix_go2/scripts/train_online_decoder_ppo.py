#!/usr/bin/env python3
"""Online decoder-only PPO fine-tuning for Vintix (train.py 風エントリ)."""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
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

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

GENESIS_LOCOMOTION = Path(__file__).resolve().parents[2] / "Genesis" / "examples" / "locomotion"
sys.path.insert(0, str(GENESIS_LOCOMOTION))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import genesis as gs
from env import A1Env, ANYmalCEnv, Go1Env, Go2Env, MiniCheetahEnv
from train import (  # noqa: E402
    get_anymalc_cfgs,
    get_go1_cfgs,
    get_go2_cfgs,
    get_minicheetah_cfgs,
    get_train_cfg,
    get_unitreea1_cfgs,
    resolve_robot_type,
)

from vintix_actor_critic import VintixActorCritic, load_vintix_ckpt_defaults  # noqa: E402


ROBOT_TASK_NAME = {
    "go2": "go2_walking_ad",
    "go1": "go1_walking_ad",
    "unitreea1": "a1_walking_ad",
    "a1": "a1_walking_ad",
    "minicheetah": "minicheetah_walking_ad",
    "anymalc": "anymalc_walking_ad",
}


class FeatureStoragePPO(PPO):
    """PPO: ロールアウト時に actor obs の代わりに policy features を保存する。"""

    def __init__(self, actor_critic, **kwargs):
        # 親が全パラメータで Adam を作る前に、trainable のみ渡すため一旦差し替え
        super().__init__(actor_critic, **kwargs)
        trainable = [p for p in actor_critic.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable, lr=self.learning_rate)

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # ★ raw obs ではなく decoder 入力特徴を保存（PPO update で再計算可能）
        self.transition.observations = (
            self.actor_critic.get_last_policy_features().detach().clone()
        )
        self.transition.critic_observations = critic_obs
        return self.transition.actions


class OnlineDecoderOnPolicyRunner(OnPolicyRunner):
    """VintixActorCritic + FeatureStoragePPO + 履歴コミット + 報酬曲線記録。"""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = dict(train_cfg["policy"])
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs

        policy_cfg = dict(self.policy_cfg)
        policy_cfg.pop("class_name", None)
        # actor_hidden_dims は未使用だが train.py 互換のため残す
        self.alg_policy = VintixActorCritic(
            num_obs, num_critic_obs, self.env.num_actions, **policy_cfg
        ).to(self.device)

        alg_cfg = dict(self.alg_cfg)
        alg_cfg.pop("class_name", None)
        self.alg = FeatureStoragePPO(self.alg_policy, device=self.device, **alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        self.obs_normalizer = torch.nn.Identity().to(self.device)
        self.critic_obs_normalizer = torch.nn.Identity().to(self.device)

        hidden_dim = self.alg_policy.hidden_dim
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [hidden_dim],  # actor obs = policy features
            [num_critic_obs],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.logger_type = "tensorboard"
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = []
        self.reward_history = []  # list of (iter, mean_reward)

        # 履歴初期化
        self.alg_policy.seed_histories(obs.to(self.device))

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
        *,
        stop_on_saturation: bool = False,
        stop_on_death_spiral: bool = False,
        display_target_iters: int | str | None = None,
    ):
        # 親の learn をほぼ踏襲しつつ、step 後に履歴コミット・報酬記録を追加
        self.logger_type = "tensorboard"
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()

        from collections import deque
        import time

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        display_tot = display_target_iters if display_target_iters is not None else tot_iter
        stopped_by_saturation = False
        stopped_by_death_spiral = False
        # 判定は「この learn() 呼び出しで得た統計」のみ（resume 済み履歴は混ぜない）
        run_rewards: list[float] = []
        run_ep_lens: list[float] = []
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Vintix SAR 履歴を更新
                    self.alg_policy.commit_transition(obs, actions, rewards, dones)

                    if "critic" in infos["observations"]:
                        critic_obs = infos["observations"]["critic"].to(self.device)
                    else:
                        critic_obs = obs

                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss = (
                self.alg.update()
            )
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            mean_rew = float(np.mean(rewbuffer)) if len(rewbuffer) > 0 else 0.0
            mean_len = float(np.mean(lenbuffer)) if len(lenbuffer) > 0 else 0.0
            self.reward_history.append((it, mean_rew))
            run_rewards.append(mean_rew)
            run_ep_lens.append(mean_len)
            self.tot_timesteps += self.env.num_envs * self.num_steps_per_env
            self.tot_time += collection_time + learn_time

            if self.writer is not None:
                self.writer.add_scalar("Train/mean_reward", mean_rew, it)
                self.writer.add_scalar("Train/mean_episode_length", mean_len, it)
                self.writer.add_scalar("Loss/value_function", mean_value_loss, it)
                self.writer.add_scalar("Loss/surrogate", mean_surrogate_loss, it)
                self.writer.add_scalar("Loss/entropy", mean_entropy, it)

            fps = int(self.env.num_envs * self.num_steps_per_env / max(collection_time + learn_time, 1e-6))
            # 絶対 1-based（再開しても model_N / 表示が連続する）
            epoch_1based = it + 1
            print(
                f"iter {epoch_1based}/{display_tot}  mean_reward={mean_rew:.3f}  "
                f"mean_ep_len={mean_len:.1f}  "
                f"v_loss={mean_value_loss:.4f}  surr={mean_surrogate_loss:.4f}  "
                f"fps={fps}  collect={collection_time:.2f}s  learn={learn_time:.2f}s"
            )

            # 即死 → 成功飽和の順（閾値はモジュール定数）
            if stop_on_death_spiral and not stopped_by_death_spiral:
                is_death, death_msg = death_spiral_detected(run_ep_lens)
                if is_death:
                    print(f"[death] stop at iter {epoch_1based}: {death_msg}")
                    stopped_by_death_spiral = True

            if stop_on_saturation and not stopped_by_death_spiral:
                saturated, sat_msg = reward_saturated(run_rewards)
                if saturated:
                    print(f"[saturate] stop at iter {epoch_1based}: {sat_msg}")
                    stopped_by_saturation = True

            # 絶対 1-based: save_interval ごと + 最終 / 早期停止時は必ず保存
            early_stop = stopped_by_saturation or stopped_by_death_spiral
            is_last = (it + 1 == tot_iter) or early_stop
            should_save = self.log_dir is not None and (
                (epoch_1based % self.save_interval == 0) or is_last
            )
            if should_save:
                tag = f"model_{epoch_1based}"
                ckpt_path = os.path.join(self.log_dir, f"{tag}.pt")
                self.save(ckpt_path)
                export_path = Path(self.log_dir) / tag
                export_path.mkdir(parents=True, exist_ok=True)
                self.alg_policy.export_vintix(str(export_path))
                print(f"[checkpoint] saved {tag}.pt and {tag}/ (Vintix)")

            ep_infos.clear()
            if early_stop:
                break

        # 最終は上の should_save で保存済み（重複保存しない）
        return {
            "stopped_by_saturation": stopped_by_saturation,
            "stopped_by_death_spiral": stopped_by_death_spiral,
            "last_iter_1based": self.current_learning_iteration + 1,
        }

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

    def load(self, path: str, load_optimizer: bool = True):
        loaded = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded:
            self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        # 保存値は完了済み 0-based index → 次に回す iter は +1
        self.current_learning_iteration = int(loaded.get("iter", -1)) + 1
        return loaded.get("infos")

    def load_reward_history_csv(self, csv_path: Path) -> None:
        """既存の学習曲線 CSV を reward_history に読み込む（再開時の結合用）。"""
        if not csv_path.is_file():
            return
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = [(int(r["iteration"]), float(r["mean_reward"])) for r in reader]
        if not rows:
            return
        self.reward_history = rows
        print(f"[resume] loaded {len(rows)} reward history points from {csv_path}")


def _build_env(robot_type: str, num_envs: int, device):
    cfgs = {
        "go2": get_go2_cfgs,
        "go1": get_go1_cfgs,
        "unitreea1": get_unitreea1_cfgs,
        "minicheetah": get_minicheetah_cfgs,
        "anymalc": get_anymalc_cfgs,
    }
    if robot_type not in cfgs:
        raise ValueError(f"Unsupported robot_type: {robot_type}")
    env_cfg, obs_cfg, reward_cfg, command_cfg = cfgs[robot_type]()
    env_cls = {
        "go2": Go2Env,
        "go1": Go1Env,
        "unitreea1": A1Env,
        "minicheetah": MiniCheetahEnv,
        "anymalc": ANYmalCEnv,
    }[robot_type]
    env = env_cls(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        domain_randomization=False,
        mass_range=(1.0, 1.0),
    )
    return env, env_cfg, obs_cfg, reward_cfg, command_cfg


TRAIN_REWARD_Y_LIM = (-0.2, 25.0)
TRAIN_FONT_LABEL = 34
TRAIN_FONT_TICK = 28

# 報酬飽和早期終了の内部既定（CLI では --saturate のみ）
# 相対％は |mean|≈0 の即死ループで発火しないため、絶対幅 ε を使う。
# WINDOW 本の max-min 自体が「複数 iter の平坦さ」なので、連続カウントは不要。
SATURATE_WINDOW = 20  # 直近この本数の mean_reward で max-min を見る
SATURATE_EPS = 0.25  # max(window)-min(window) <= EPS なら飽和候補
SATURATE_MIN_MEAN_REWARD = 1.0  # これ未満の平坦は成功飽和とみなさない（即死と分離）
SATURATE_MIN_RUN_ITERS = 1000  # 序盤の一時平坦（Go2 成功例 ~600 iter）を誤検知しにくくする

# 即死ループ検知（短エピソードが続く失敗モード）
DEATH_WINDOW = 20
DEATH_MAX_MEAN_EP_LEN = 30  # MiniCheetah 失敗 ~6、歩行は数百〜1000
DEATH_MIN_RUN_ITERS = 1000
DEFAULT_INIT_NOISE_STD = 0.2
UNLIMITED_MAX_ITERATIONS = 10**9
EXIT_DEATH_SPIRAL = 2


def reward_saturated(run_rewards: list[float]) -> tuple[bool, str]:
    """直近 WINDOW の報酬レンジ ≤ ε かつ平均報酬が十分高いとき成功飽和。"""
    if len(run_rewards) < max(SATURATE_WINDOW, SATURATE_MIN_RUN_ITERS):
        return False, ""
    window = run_rewards[-SATURATE_WINDOW:]
    w_max = float(np.max(window))
    w_min = float(np.min(window))
    w_mean = float(np.mean(window))
    abs_range = w_max - w_min
    if abs_range > SATURATE_EPS:
        return False, ""
    if w_mean < SATURATE_MIN_MEAN_REWARD:
        return False, ""
    msg = (
        f"window={SATURATE_WINDOW}, abs_range={abs_range:.4f} <= eps={SATURATE_EPS}, "
        f"mean={w_mean:.4f} >= min_mean={SATURATE_MIN_MEAN_REWARD} "
        f"(min={w_min:.4f}, max={w_max:.4f})"
    )
    return True, msg


def death_spiral_detected(run_ep_lens: list[float]) -> tuple[bool, str]:
    """直近 WINDOW の平均 ep_len が閾値未満なら即死ループ。"""
    if len(run_ep_lens) < max(DEATH_WINDOW, DEATH_MIN_RUN_ITERS):
        return False, ""
    window = run_ep_lens[-DEATH_WINDOW:]
    mean_len = float(np.mean(window))
    if mean_len >= DEATH_MAX_MEAN_EP_LEN:
        return False, ""
    msg = (
        f"window={DEATH_WINDOW}, mean_ep_len={mean_len:.2f} < "
        f"max={DEATH_MAX_MEAN_EP_LEN}"
    )
    return True, msg


def _attempts_path(log_dir: Path) -> Path:
    return log_dir / "seed_attempts.json"


def _append_attempt(log_dir: Path, record: dict) -> None:
    import json

    path = _attempts_path(log_dir)
    data: dict = {"attempts": []}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {"attempts": []}
    data.setdefault("attempts", []).append(record)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _should_keep_on_archive(name: str) -> bool:
    return name.startswith("failed_seed") or name in {
        "seed_attempts.json",
        "run_config.json",
    }


def archive_failed_attempt(log_dir: Path, result_dir: Path, seed: int) -> Path:
    """失敗試行の成果物を failed_seed{S}/ へ退避（成功試行が直下に残るようにする）。"""
    import shutil

    dest_log = log_dir / f"failed_seed{seed}"
    dest_log.mkdir(parents=True, exist_ok=True)
    for p in list(log_dir.iterdir()):
        if p.name == dest_log.name or _should_keep_on_archive(p.name):
            continue
        target = dest_log / p.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(p), str(target))

    if result_dir.resolve() != log_dir.resolve():
        dest_res = result_dir / f"failed_seed{seed}"
        dest_res.mkdir(parents=True, exist_ok=True)
        for name in ("train_mean_reward.csv", "train_mean_reward.png"):
            src = result_dir / name
            if src.is_file():
                shutil.move(str(src), str(dest_res / name))
    print(f"[death] archived attempt seed={seed} → {dest_log}")
    return dest_log


def _replace_argv_flag(argv: list[str], flag: str, value: str) -> list[str]:
    out = list(argv)
    if flag in out:
        i = out.index(flag)
        if i + 1 < len(out) and not out[i + 1].startswith("-"):
            out[i + 1] = value
        else:
            out.insert(i + 1, value)
    else:
        out.extend([flag, value])
    return out


def _spawn_seed_retry(next_seed: int, remaining_retries: int) -> int:
    """Genesis 再生成を避けるため、同一スクリプトを新しい seed で subprocess 起動。"""
    import subprocess

    child_argv = _replace_argv_flag(sys.argv[1:], "--seed", str(next_seed))
    child_argv = _replace_argv_flag(child_argv, "--seed_retries", str(remaining_retries))
    cmd = [sys.executable, str(Path(__file__).resolve()), *child_argv]
    print(
        f"[death] retrying with seed={next_seed}, seed_retries_left={remaining_retries}\n"
        f"        cmd={' '.join(cmd)}",
        flush=True,
    )
    return int(subprocess.call(cmd))


def save_reward_curve(
    reward_history,
    out_csv: Path,
    out_png: Path,
    *,
    num_envs: int,
    num_steps_per_env: int,
) -> None:
    """Readable スタイル: 横軸=環境ステップ数（科学表記）, 縦軸=平均報酬, ylim 固定。"""
    from matplotlib.ticker import ScalarFormatter
    from plot_style import FIGSIZE, LINE_WIDTH

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "env_steps", "mean_reward"])
        for it, rew in reward_history:
            steps = (int(it) + 1) * num_envs * num_steps_per_env
            writer.writerow([it, steps, rew])
    if not reward_history:
        return
    steps = [(int(it) + 1) * num_envs * num_steps_per_env for it, _ in reward_history]
    rews = [rew for _, rew in reward_history]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(steps, rews, linewidth=LINE_WIDTH, color="#ff7f0e")
    ax.set_xlabel("step", fontsize=TRAIN_FONT_LABEL)
    ax.set_ylabel("reward", fontsize=TRAIN_FONT_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=TRAIN_FONT_TICK)
    ax.set_ylim(*TRAIN_REWARD_Y_LIM)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4), useMathText=True)
    ax.grid(True, alpha=0.3)
    fig.canvas.draw()
    ax.xaxis.get_offset_text().set_fontsize(TRAIN_FONT_TICK)
    ax.yaxis.get_offset_text().set_fontsize(TRAIN_FONT_TICK)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reward curve: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Online decoder-only PPO FT for Vintix")
    parser.add_argument("-e", "--exp_name", type=str, default="online_decoder_ft")
    parser.add_argument("-r", "--robot_type", type=str, required=True)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="Hard cap on absolute PPO iterations (1-based target). "
        "Use -1 with --saturate to run until reward saturates (no practical hard cap).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--seed_retries",
        type=int,
        default=0,
        help="On death-spiral stop, archive and retry with seed+1 up to N more times "
        "(recommended 3). Incompatible with --resume_path.",
    )
    parser.add_argument("--vintix_path", type=str, required=True, help="Vintix epoch directory")
    parser.add_argument("--log_dir", type=str, required=True, help="PPO/Vintix checkpoints (model_N.pt + model_N/)")
    parser.add_argument("--result_dir", type=str, default=None, help="Training curve CSV/PNG (default: log_dir)")
    parser.add_argument(
        "--context_len",
        type=int,
        default=None,
        help="SAR history length (default: read from ckpt config.json, typically 2048)",
    )
    parser.add_argument(
        "--trajectory_sparsity",
        type=int,
        default=None,
        help="Must match pretraining (default: read from ckpt config.json, typically 128)",
    )
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save PPO/Vintix checkpoint every N PPO iterations (1-based absolute), plus final iter",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Resume from PPO checkpoint model_N.pt (continues absolute iter numbering)",
    )
    parser.add_argument(
        "--init_noise_std",
        type=float,
        default=DEFAULT_INIT_NOISE_STD,
        help=f"Initial Gaussian action noise std (default {DEFAULT_INIT_NOISE_STD}). "
        "Smaller than train.py's 1.0 to avoid early fall collapse during decoder FT.",
    )
    parser.add_argument(
        "--saturate",
        action="store_true",
        help="Stop early when mean_reward saturates (built-in window/threshold). "
        "Keeps --max_iterations as hard cap unless -1.",
    )
    parser.add_argument("--clear_history_on_done", action="store_true")
    args = parser.parse_args()

    if args.seed_retries < 0:
        raise ValueError(f"--seed_retries must be >= 0, got {args.seed_retries}")
    if args.resume_path and args.seed_retries > 0:
        raise ValueError("--seed_retries is incompatible with --resume_path")

    if args.max_iterations == -1:
        if not args.saturate:
            raise ValueError("--max_iterations -1 requires --saturate")
        args.max_iterations = UNLIMITED_MAX_ITERATIONS
        unlimited_by_saturate = True
    else:
        unlimited_by_saturate = False
    if args.max_iterations < 1:
        raise ValueError(f"--max_iterations must be >=1 or -1, got {args.max_iterations}")

    robot_type = resolve_robot_type(args.robot_type)
    task_name = args.task_name or ROBOT_TASK_NAME.get(robot_type)
    if not task_name:
        raise ValueError(f"Cannot resolve task_name for robot {robot_type}")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    ckpt_defaults = load_vintix_ckpt_defaults(args.vintix_path)
    context_len = args.context_len if args.context_len is not None else ckpt_defaults["context_len"]
    trajectory_sparsity = (
        args.trajectory_sparsity
        if args.trajectory_sparsity is not None
        else ckpt_defaults["trajectory_sparsity"]
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(args.result_dir) if args.result_dir else log_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, seed=args.seed)
    train_cfg["save_interval"] = args.save_interval
    train_cfg["policy"] = {
        "activation": "elu",
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "init_noise_std": args.init_noise_std,
        "class_name": "VintixActorCritic",
        "vintix_path": args.vintix_path,
        "task_name": task_name,
        "group_name": "quadruped_locomotion",
        "context_len": context_len,
        "trajectory_sparsity": trajectory_sparsity,
        "clear_history_on_done": args.clear_history_on_done,
    }
    # FeatureStoragePPO の storage は hidden_dim 幅。algorithm はそのまま。

    env, env_cfg, obs_cfg, reward_cfg, command_cfg = _build_env(
        robot_type, args.num_envs, gs.device
    )
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(log_dir / "cfgs.pkl", "wb"),
    )

    runner = OnlineDecoderOnPolicyRunner(
        env, train_cfg, str(log_dir), device=gs.device
    )

    reward_csv = result_dir / "train_mean_reward.csv"
    stop_on_death = args.resume_path is None
    if args.resume_path:
        resume_path = Path(args.resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume_path not found: {resume_path}")
        runner.load(str(resume_path))
        runner.load_reward_history_csv(reward_csv)
        remaining = args.max_iterations - runner.current_learning_iteration
        if remaining <= 0:
            raise ValueError(
                f"Nothing to train: resume starts at iter {runner.current_learning_iteration + 1} "
                f"but max_iterations={args.max_iterations}"
            )
        print(
            f"Resume from {resume_path}: next iter={runner.current_learning_iteration + 1}, "
            f"remaining={remaining}, target={args.max_iterations}"
        )
    else:
        remaining = args.max_iterations

    display_target = "saturate" if unlimited_by_saturate else str(args.max_iterations)
    print(
        f"Start online decoder PPO: robot={robot_type}, task={task_name}, "
        f"envs={args.num_envs}, target_iters={display_target}, "
        f"this_run={remaining}, seed={args.seed}, ckpt={args.vintix_path}"
    )
    print(
        f"Inference config: context_len={context_len}, trajectory_sparsity={trajectory_sparsity} "
        f"(from ckpt unless overridden)"
    )
    print(f"Policy init_noise_std={args.init_noise_std}")
    if args.saturate:
        print(
            f"Saturation stop enabled "
            f"(window={SATURATE_WINDOW}, abs_range<=eps={SATURATE_EPS}, "
            f"min_mean_reward={SATURATE_MIN_MEAN_REWARD}, "
            f"min_run_iters={SATURATE_MIN_RUN_ITERS})"
        )
    if stop_on_death:
        print(
            f"Death-spiral stop enabled "
            f"(window={DEATH_WINDOW}, mean_ep_len<{DEATH_MAX_MEAN_EP_LEN}, "
            f"min_run_iters={DEATH_MIN_RUN_ITERS}, seed_retries={args.seed_retries})"
        )
    print(
        f"Checkpoints: every {args.save_interval} absolute iters + final → "
        f"model_N.pt + model_N/ under {log_dir}"
    )
    learn_info = runner.learn(
        num_learning_iterations=remaining,
        init_at_random_ep_len=True,
        stop_on_saturation=args.saturate,
        stop_on_death_spiral=stop_on_death,
        display_target_iters="saturate" if unlimited_by_saturate else args.max_iterations,
    )

    if runner.writer is not None:
        runner.writer.flush()
        runner.writer.close()
        runner.writer = None

    save_reward_curve(
        runner.reward_history,
        reward_csv,
        result_dir / "train_mean_reward.png",
        num_envs=args.num_envs,
        num_steps_per_env=train_cfg["num_steps_per_env"],
    )

    from datetime import datetime, timezone

    last_1based = int(learn_info["last_iter_1based"])
    attempt_record = {
        "seed": args.seed,
        "status": (
            "death_spiral"
            if learn_info.get("stopped_by_death_spiral")
            else (
                "saturated"
                if learn_info.get("stopped_by_saturation")
                else "completed"
            )
        ),
        "last_iter": last_1based,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    if learn_info.get("stopped_by_death_spiral"):
        archive_failed_attempt(log_dir, result_dir, args.seed)
        _append_attempt(log_dir, attempt_record)
        if args.seed_retries > 0:
            rc = _spawn_seed_retry(args.seed + 1, args.seed_retries - 1)
            raise SystemExit(rc)
        print(
            f"[death] no seed retries left (seed={args.seed}). "
            f"See {_attempts_path(log_dir)}",
            flush=True,
        )
        raise SystemExit(EXIT_DEATH_SPIRAL)

    _append_attempt(log_dir, attempt_record)

    final_vintix = log_dir / f"model_{last_1based}"
    if not (final_vintix / "model.pth").is_file():
        # max_iterations が save_interval の倍数でない場合のフォールバック
        candidates = sorted(log_dir.glob("model_*/model.pth"), key=lambda p: p.parent.name)
        if candidates:
            print(f"Final Vintix export OK: {candidates[-1].parent}")
        else:
            raise FileNotFoundError(f"No Vintix export under {log_dir}")
    else:
        print(f"Final Vintix export OK: {final_vintix}")
    if learn_info.get("stopped_by_saturation"):
        print(f"Stopped by reward saturation at model_{last_1based}")


if __name__ == "__main__":
    main()
