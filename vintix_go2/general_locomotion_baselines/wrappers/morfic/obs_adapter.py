"""Map Genesis loco obs (45) → MorFiC-style single-step (70) + history (30×70).

MorFiC train config (scripts/train.py + pretrained ac_weights_last.pt):
  num_observations = 70
  num_observation_history = 30
  → adaptation_module input 2100

Layout of the 70-dim vector (observe_command + two_prev_actions + clock):
  projected_gravity (3)
  + commands (15)          # Genesis only exposes 3 velocity cmds; gait slots padded
  + dof_pos_rel (12)
  + dof_vel (12)
  + actions (12)
  + last_actions (12)
  + clock_inputs (4)
  = 70

This is an *approximate* reconstruction for wiring / shape tests.
Binary parity with Isaac Gym MorFiC is not claimed (see NOTES.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Genesis go*_env obs layout (num_obs=45)
GENESIS_SLICES = {
    "ang_vel": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "commands": slice(6, 9),
    "dof_pos": slice(9, 21),
    "dof_vel": slice(21, 33),
    "actions": slice(33, 45),
}

MORFIC_OBS_DIM = 70
MORFIC_HISTORY_LEN = 30
MORFIC_HISTORY_FLAT = MORFIC_OBS_DIM * MORFIC_HISTORY_LEN  # 2100
NUM_COMMANDS = 15
CLOCK_DIM = 4


@dataclass
class GenesisToMorficObs:
    """Stateful history buffer: call `reset` then `step` each control tick."""

    batch_size: int = 1
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32
    history_len: int = MORFIC_HISTORY_LEN
    obs_dim: int = MORFIC_OBS_DIM

    def __post_init__(self):
        self.device = torch.device(self.device)
        self._history = torch.zeros(
            self.batch_size,
            self.history_len,
            self.obs_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self._last_actions = torch.zeros(
            self.batch_size, 12, device=self.device, dtype=self.dtype
        )
        self._clock_phase = torch.zeros(
            self.batch_size, device=self.device, dtype=self.dtype
        )

    def reset(self, batch_idx: slice | None = None):
        if batch_idx is None:
            self._history.zero_()
            self._last_actions.zero_()
            self._clock_phase.zero_()
        else:
            self._history[batch_idx].zero_()
            self._last_actions[batch_idx].zero_()
            self._clock_phase[batch_idx].zero_()

    def genesis_to_single(self, genesis_obs: torch.Tensor) -> torch.Tensor:
        """Convert (B, 45) Genesis obs → (B, 70) MorFiC-like obs."""
        if genesis_obs.ndim != 2 or genesis_obs.shape[-1] != 45:
            raise ValueError(f"expected (B, 45) Genesis obs, got {tuple(genesis_obs.shape)}")
        b = genesis_obs.shape[0]
        if b != self.batch_size:
            raise ValueError(f"batch {b} != adapter batch_size {self.batch_size}")

        g = genesis_obs.to(device=self.device, dtype=self.dtype)
        gravity = g[:, GENESIS_SLICES["projected_gravity"]]
        cmd3 = g[:, GENESIS_SLICES["commands"]]
        dof_pos = g[:, GENESIS_SLICES["dof_pos"]]
        dof_vel = g[:, GENESIS_SLICES["dof_vel"]]
        actions = g[:, GENESIS_SLICES["actions"]]

        commands = torch.zeros(b, NUM_COMMANDS, device=self.device, dtype=self.dtype)
        commands[:, :3] = cmd3
        # Remaining gait command slots stay 0 (approx; official uses phase/freq/etc.)

        # Cheap synthetic foot clocks (not the official gait index); keeps dim=4 filled.
        phase = self._clock_phase
        clock = torch.stack(
            [
                torch.sin(2 * torch.pi * phase),
                torch.sin(2 * torch.pi * (phase + 0.5)),
                torch.sin(2 * torch.pi * phase),
                torch.sin(2 * torch.pi * (phase + 0.5)),
            ],
            dim=-1,
        )
        self._clock_phase = (phase + 0.02) % 1.0

        single = torch.cat(
            [gravity, commands, dof_pos, dof_vel, actions, self._last_actions, clock],
            dim=-1,
        )
        if single.shape[-1] != self.obs_dim:
            raise RuntimeError(f"built obs dim {single.shape[-1]} != {self.obs_dim}")

        self._last_actions = actions.detach().clone()
        return single

    def step(self, genesis_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Push one Genesis frame; return single + flat history."""
        single = self.genesis_to_single(genesis_obs)
        self._history = torch.cat(
            [self._history[:, 1:], single.unsqueeze(1)], dim=1
        )
        flat = self._history.reshape(self.batch_size, -1)
        return {
            "obs": single,
            "obs_history": self._history.clone(),
            "obs_history_flat": flat,
        }
