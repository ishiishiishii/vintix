"""rsl_rl 互換の Vintix actor + MLP critic（online decoder-only PPO 用）。"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

from vintix.vintix import Vintix


PROPRIO_DIM = 33  # full env obs is 45 = proprio(33) + last_action(12)

# 事前学習 Vintix の既定（*_without ckpt の config.json と一致）
DEFAULT_CONTEXT_LEN = 2048
DEFAULT_TRAJECTORY_SPARSITY = 128


def load_vintix_ckpt_defaults(vintix_path: str) -> dict:
    """ckpt の config.json から context_len / trajectory_sparsity を読む。"""
    import json
    from pathlib import Path

    config_path = Path(vintix_path) / "config.json"
    if not config_path.is_file():
        return {
            "context_len": DEFAULT_CONTEXT_LEN,
            "trajectory_sparsity": DEFAULT_TRAJECTORY_SPARSITY,
        }
    with open(config_path, "r", encoding="utf-8") as f:
        conf = json.load(f)
    return {
        "context_len": int(conf.get("context_len", DEFAULT_CONTEXT_LEN)),
        "trajectory_sparsity": int(
            conf.get("trajectory_sparsity", DEFAULT_TRAJECTORY_SPARSITY)
        ),
    }


def compute_averaged_stats(model_metadata: dict, task_name: str) -> dict:
    """未知タスク用: 既知タスクの obs/acs 正規化を平均（save_vintix と同趣旨）。"""
    obs_means, obs_stds, acs_means, acs_stds = [], [], [], []
    for _tn, meta in model_metadata.items():
        if all(k in meta for k in ("obs_mean", "obs_std", "acs_mean", "acs_std")):
            obs_means.append(np.array(meta["obs_mean"], dtype=np.float64))
            obs_stds.append(np.array(meta["obs_std"], dtype=np.float64))
            acs_means.append(np.array(meta["acs_mean"], dtype=np.float64))
            acs_stds.append(np.array(meta["acs_std"], dtype=np.float64))
    if not obs_means:
        raise ValueError("No known-task normalization stats in model metadata.")
    return {
        task_name: {
            "obs_mean": np.mean(obs_means, axis=0).tolist(),
            "obs_std": np.mean(obs_stds, axis=0).tolist(),
            "acs_mean": np.mean(acs_means, axis=0).tolist(),
            "acs_std": np.mean(acs_stds, axis=0).tolist(),
        }
    }


def freeze_encoder_transformer_except_decoder(model: Vintix, decoder_group: str) -> None:
    """encoder / transformer / 他 decoder を freezeし、指定 group の AcsDecoder のみ更新可能にする。"""
    for p in model.transformer.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = False
    for gn, dec in model.head.acs_decoders.items():
        for p in dec.parameters():
            p.requires_grad = gn == decoder_group


class _EnvHistory:
    """1 env 分の SAR 履歴（エピソード跨ぎ保持）。"""

    def __init__(self, max_len: int, task_name: str):
        self.max_len = max_len
        self.task_name = task_name
        self.observations: Deque[np.ndarray] = deque(maxlen=max_len)
        self.actions: Deque[np.ndarray] = deque(maxlen=max_len)
        self.rewards: Deque[float] = deque(maxlen=max_len)
        self.step_nums: Deque[int] = deque(maxlen=max_len)
        self.current_step = 0

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.step_nums.clear()
        self.current_step = 0

    def seed(self, obs_proprio: np.ndarray) -> None:
        self.clear()
        self.add(obs_proprio, np.zeros(12, dtype=np.float32), 0.0)

    def add(self, obs_proprio: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.observations.append(np.asarray(obs_proprio, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        self.step_nums.append(int(self.current_step))
        self.current_step += 1

    def as_batch_item(self, device: torch.device) -> Dict[str, Any]:
        obs = torch.tensor(np.asarray(self.observations), dtype=torch.float32, device=device)
        act = torch.tensor(np.asarray(self.actions), dtype=torch.float32, device=device)
        rew = torch.tensor(np.asarray(self.rewards), dtype=torch.float32, device=device).unsqueeze(1)
        step = torch.tensor(np.asarray(self.step_nums), dtype=torch.int32, device=device)
        return {
            "observation": obs,
            "prev_action": act,
            "prev_reward": rew,
            "step_num": step,
            "task_name": self.task_name,
        }


class VintixActorCritic(nn.Module):
    """Vintix (decoder-only trainable) actor + train.py 同型 MLP critic。"""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=None,  # unused (Vintix is the actor)
        critic_hidden_dims=None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        vintix_path: str = "",
        task_name: str = "go2_walking_ad",
        group_name: str = "quadruped_locomotion",
        context_len: Optional[int] = None,
        trajectory_sparsity: Optional[int] = None,
        clear_history_on_done: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "VintixActorCritic.__init__ ignoring unexpected kwargs: "
                + str(list(kwargs.keys()))
            )
        super().__init__()
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]
        if not vintix_path:
            raise ValueError("vintix_path is required for VintixActorCritic")

        self.num_actor_obs = int(num_actor_obs)
        self.num_actions = int(num_actions)
        self.task_name = task_name
        self.group_name = group_name
        self.clear_history_on_done = bool(clear_history_on_done)
        self.proprio_dim = PROPRIO_DIM

        # --- Vintix actor ---
        self.vintix = Vintix()
        self.vintix.load_model(vintix_path)
        ckpt_defaults = load_vintix_ckpt_defaults(vintix_path)
        self.context_len = int(
            context_len if context_len is not None else ckpt_defaults["context_len"]
        )
        self.trajectory_sparsity = int(
            trajectory_sparsity
            if trajectory_sparsity is not None
            else ckpt_defaults["trajectory_sparsity"]
        )
        if task_name not in self.vintix.head.task2group:
            stats = compute_averaged_stats(self.vintix.metadata, task_name)
            self.vintix.add_task(task_name, group_name, stats, rew_scale=1.0)
            print(f"[VintixActorCritic] Added unknown task {task_name} → group {group_name}")
        self.decoder_group = self.vintix.head.task2group[task_name]
        freeze_encoder_transformer_except_decoder(self.vintix, self.decoder_group)
        self.hidden_dim = int(self.vintix.conf["hidden_dim"])

        # --- MLP critic (train.py と同型) ---
        act_fn = resolve_nn_activation(activation)
        critic_layers: List[nn.Module] = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(act_fn)
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                critic_layers.append(act_fn)
        self.critic = nn.Sequential(*critic_layers)

        # --- action noise ---
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution: Optional[Normal] = None
        Normal.set_default_validate_args(False)

        self._histories: List[_EnvHistory] = []
        self._last_policy_features: Optional[torch.Tensor] = None
        self._last_actions: Optional[torch.Tensor] = None

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[VintixActorCritic] trainable={trainable:,} / {total:,} "
            f"({100.0 * trainable / max(total, 1):.2f}%), decoder_group={self.decoder_group}, "
            f"context_len={self.context_len}, trajectory_sparsity={self.trajectory_sparsity}"
        )
        # backbone は常に eval（dropout 無効）。decoder / critic / std は train() で切替。
        self.vintix.encoder.eval()
        self.vintix.transformer.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vintix.encoder.eval()
        self.vintix.transformer.eval()
        return self

    def _ensure_histories(self, num_envs: int) -> None:
        while len(self._histories) < num_envs:
            self._histories.append(_EnvHistory(self.context_len, self.task_name))

    def seed_histories(self, obs: torch.Tensor) -> None:
        """学習開始時に各 env の履歴を (obs, 0, 0) で初期化。"""
        obs_np = obs.detach().cpu().numpy()
        n = obs_np.shape[0]
        self._ensure_histories(n)
        for i in range(n):
            proprio = obs_np[i, : self.proprio_dim]
            self._histories[i].seed(proprio)

    def commit_transition(
        self,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """env.step 後に履歴へ (next_obs, action, reward) を追加。"""
        next_np = next_obs.detach().cpu().numpy()
        act_np = actions.detach().cpu().numpy()
        rew_np = rewards.detach().cpu().numpy()
        done_np = dones.detach().cpu().numpy()
        n = next_np.shape[0]
        self._ensure_histories(n)
        for i in range(n):
            if self.clear_history_on_done and bool(done_np[i]):
                # リセット後の観測で履歴をやり直す
                self._histories[i].seed(next_np[i, : self.proprio_dim])
            else:
                self._histories[i].add(
                    next_np[i, : self.proprio_dim],
                    act_np[i],
                    float(rew_np[i]),
                )

    def get_last_policy_features(self) -> torch.Tensor:
        if self._last_policy_features is None:
            raise RuntimeError("get_last_policy_features called before act()")
        return self._last_policy_features

    def _decode_mean_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """(B, hidden) → (B, action_dim)。decoder のみ（勾配あり）。"""
        decoder = self.vintix.head.acs_decoders[self.decoder_group]
        # AcsDecoder expects (traj_len, hidden) or (B, hidden); Linear works on last dim
        mean = decoder(features)
        if self.vintix.head.unnorm_acs:
            task_meta = self.vintix.head.task_metadata[self.task_name]
            acs_mean = torch.tensor(task_meta["acs_mean"], dtype=mean.dtype, device=mean.device)
            acs_std = torch.tensor(task_meta["acs_std"], dtype=mean.dtype, device=mean.device)
            mean = mean * acs_std + acs_mean
        return mean

    def _forward_features_from_history(self) -> torch.Tensor:
        """全 env の履歴をバッチ forwardし、最終 token 特徴 (N, hidden) を返す（backbone no_grad）。"""
        device = next(self.parameters()).device
        batch = [h.as_batch_item(device) for h in self._histories]
        # 長さが揃っている前提（全 env 同時開始・履歴クリアなし）。ズレていれば最短に揃える。
        lengths = [b["observation"].shape[0] for b in batch]
        min_len = min(lengths)
        if len(set(lengths)) != 1:
            for b in batch:
                for key in ("observation", "prev_action", "prev_reward", "step_num"):
                    b[key] = b[key][-min_len:]

        with torch.no_grad():
            embs = self.vintix.encoder.forward(batch)
            # encoder/transformer は freeze。特徴は定数として decoder に渡す。
            tokens = embs["input_tokens"]
            # context_len 上限でクリップ（事前学習と同じ 2048）
            if tokens.shape[1] > self.context_len:
                tokens = tokens[:, -self.context_len :, :]
            trans_out = self.vintix.transformer(tokens)
            last_feat = trans_out[:, -1, :].float()
        # inference_mode 外でも使える定数特徴にする
        return last_feat.detach().clone()

    def reset(self, dones=None):
        # 既定は履歴保持。clear_history_on_done 時は commit_transition 側で処理。
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _std_tensor(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return self.std.expand_as(mean)
        return torch.exp(self.log_std).expand_as(mean)

    def update_distribution_from_features(self, features: torch.Tensor) -> None:
        mean = self._decode_mean_from_features(features)
        std = self._std_tensor(mean)
        self.distribution = Normal(mean, std)

    def update_distribution(self, observations: torch.Tensor) -> None:
        """PPO update 用: observations が (B, hidden_dim) の policy features。"""
        if observations.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                f"Expected policy features of dim {self.hidden_dim}, got {observations.shape}"
            )
        self.update_distribution_from_features(observations)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        - ロールアウト: obs が (N, num_actor_obs) → 履歴 forward → sample
        - PPO update: obs が (B, hidden_dim) → decoder only → sample（log_prob 再計算用）
        """
        if observations.shape[-1] == self.hidden_dim:
            self.update_distribution_from_features(observations)
            return self.distribution.sample()

        # rollout path
        n = observations.shape[0]
        self._ensure_histories(n)
        # 履歴未初期化なら現在観測で seed
        for i, h in enumerate(self._histories[:n]):
            if len(h.observations) == 0:
                proprio = observations[i, : self.proprio_dim].detach().cpu().numpy()
                h.seed(proprio)

        features = self._forward_features_from_history()
        self._last_policy_features = features
        self.update_distribution_from_features(features)
        actions = self.distribution.sample()
        self._last_actions = actions.detach()
        return actions

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] == self.hidden_dim:
            return self._decode_mean_from_features(observations)
        n = observations.shape[0]
        self._ensure_histories(n)
        for i, h in enumerate(self._histories[:n]):
            if len(h.observations) == 0:
                proprio = observations[i, : self.proprio_dim].detach().cpu().numpy()
                h.seed(proprio)
        features = self._forward_features_from_history()
        return self._decode_mean_from_features(features)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def export_vintix(self, path: str) -> None:
        """更新済み decoder を含む Vintix を epoch ディレクトリ形式で保存。"""
        self.vintix.save_model(path)
