"""Map Genesis loco obs (45) → GenLoco phase_only history flat (451).

Official GenLoco `build_imitation_env(enable_phase_only=True)` sensors
(`env_builder.py`) + `ImitationTask.build_target_obs(use_phase=True)`:

  MotorAngle (12) × history 15   → 180   (newest-first concat)
  IMU R,P,Y,dR,dP,dY (6) × 15    →  90
  LastAction (12) × 15           → 180
  motion phase (1)               →   1
  ────────────────────────────────────
  total                          = 451

Matches pretrained zip `model/pi_fc0/w:0` shape (451, 1024).

This is an *approximate* reconstruction for wiring / shape tests.
Binary parity with PyBullet GenLoco is not claimed (see NOTES.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GENESIS_SLICES = {
    "ang_vel": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "commands": slice(6, 9),
    "dof_pos": slice(9, 21),
    "dof_vel": slice(21, 33),
    "actions": slice(33, 45),
}

GENLOCO_HISTORY_LEN = 15
MOTOR_DIM = 12
IMU_DIM = 6  # R, P, Y, dR, dP, dY
ACTION_DIM = 12
PHASE_DIM = 1
GENLOCO_OBS_DIM = (
    MOTOR_DIM * GENLOCO_HISTORY_LEN
    + IMU_DIM * GENLOCO_HISTORY_LEN
    + ACTION_DIM * GENLOCO_HISTORY_LEN
    + PHASE_DIM
)  # 451


def _gravity_to_roll_pitch(gravity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate roll/pitch from projected gravity (g_x, g_y, g_z)."""
    gx, gy, gz = gravity[:, 0], gravity[:, 1], gravity[:, 2]
    # pitch ≈ asin(-gx) / atan2; keep finite with clamp
    pitch = torch.atan2(-gx, torch.sqrt(gy * gy + gz * gz + 1e-8))
    roll = torch.atan2(gy, gz + 1e-8)
    return roll, pitch


@dataclass
class GenesisToGenlocoObs:
    """Stateful history buffer (newest-first, matching HistoricSensorWrapper)."""

    batch_size: int = 1
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32
    history_len: int = GENLOCO_HISTORY_LEN

    def __post_init__(self):
        self.device = torch.device(self.device)
        # newest at index 0
        self._motor = torch.zeros(
            self.batch_size, self.history_len, MOTOR_DIM, device=self.device, dtype=self.dtype
        )
        self._imu = torch.zeros(
            self.batch_size, self.history_len, IMU_DIM, device=self.device, dtype=self.dtype
        )
        self._action = torch.zeros(
            self.batch_size, self.history_len, ACTION_DIM, device=self.device, dtype=self.dtype
        )
        self._phase = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)

    def reset(self, batch_idx: slice | None = None):
        if batch_idx is None:
            self._motor.zero_()
            self._imu.zero_()
            self._action.zero_()
            self._phase.zero_()
        else:
            self._motor[batch_idx].zero_()
            self._imu[batch_idx].zero_()
            self._action[batch_idx].zero_()
            self._phase[batch_idx].zero_()

    def _push(self, buf: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        # newest-first: [t, t-1, ..., t-(H-1)]
        return torch.cat([frame.unsqueeze(1), buf[:, :-1]], dim=1)

    def genesis_to_parts(self, genesis_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if genesis_obs.ndim != 2 or genesis_obs.shape[-1] != 45:
            raise ValueError(f"expected (B, 45) Genesis obs, got {tuple(genesis_obs.shape)}")
        b = genesis_obs.shape[0]
        if b != self.batch_size:
            raise ValueError(f"batch {b} != adapter batch_size {self.batch_size}")

        g = genesis_obs.to(device=self.device, dtype=self.dtype)
        motor = g[:, GENESIS_SLICES["dof_pos"]]
        gravity = g[:, GENESIS_SLICES["projected_gravity"]]
        ang = g[:, GENESIS_SLICES["ang_vel"]]
        roll, pitch = _gravity_to_roll_pitch(gravity)
        yaw = torch.zeros_like(roll)
        imu = torch.stack([roll, pitch, yaw, ang[:, 0], ang[:, 1], ang[:, 2]], dim=-1)
        action = g[:, GENESIS_SLICES["actions"]]
        return {"motor": motor, "imu": imu, "action": action}

    def step(self, genesis_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        parts = self.genesis_to_parts(genesis_obs)
        self._motor = self._push(self._motor, parts["motor"])
        self._imu = self._push(self._imu, parts["imu"])
        self._action = self._push(self._action, parts["action"])

        # Synthetic gait phase (official uses motion.calc_phase); advance ~0.02 / step.
        phase = self._phase
        self._phase = (phase + 0.02) % 1.0

        flat = torch.cat(
            [
                self._motor.reshape(self.batch_size, -1),
                self._imu.reshape(self.batch_size, -1),
                self._action.reshape(self.batch_size, -1),
                phase.unsqueeze(-1),
            ],
            dim=-1,
        )
        if flat.shape[-1] != GENLOCO_OBS_DIM:
            raise RuntimeError(f"built obs dim {flat.shape[-1]} != {GENLOCO_OBS_DIM}")
        return {
            "obs_flat": flat,
            "phase": phase,
            "motor_history": self._motor.clone(),
            "imu_history": self._imu.clone(),
            "action_history": self._action.clone(),
        }
