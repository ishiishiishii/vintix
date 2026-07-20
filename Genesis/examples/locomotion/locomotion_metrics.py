#!/usr/bin/env python3
"""Standard quadruped-locomotion evaluation metrics (scale-invariant, cross-morphology).

Motivation: cumulative / per-step reward is a fine *training* signal but a poor way to
*evaluate and compare* robots whose mass differs by >10x (the reward mixes unit-carrying
terms, so its scale is body-dependent). The locomotion literature evaluates with
dimensionless / physically grounded metrics instead. This module implements the core set
adopted for this project:

  * Cost of Transport (CoT) = Σ|τ·q̇|·dt / (m·g·d)   — energy efficiency, dimensionless.
      (Chen et al. 2023 offline-RL quadruped benchmark; Gait-Aware STL 2024;
       Hof-normalized terrain benchmark IEEE R&A-M 2025.)
  * Velocity-tracking success rate = fraction of (post-warmup) steps whose forward speed is
      within ±tol of the commanded speed. (Gait-Aware STL 2024 uses ±15%.)
  * Velocity-tracking error = RMSE of (v_xy vs cmd) and (yaw rate vs cmd).
  * Survival rate + mean episode length (fraction of envs that never fall / #steps alive).
  * Distance traveled + average forward speed (+ optional Froude number v²/gL).

Design: framework-agnostic accumulator fed per simulation step with vectorized tensors
(shape (N, ...)). All heavy state is kept as running sums so it scales to thousands of
parallel envs. Use :meth:`update_from_env` for the shared ``Go2Env``-style envs.

Reusable from both ``eval.py`` (PPO expert eval) and ``save_vintix.py`` (Vintix eval).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

DEFAULT_GRAVITY = 9.81


@dataclass
class LocomotionMetrics:
    num_envs: int
    robot_mass: float
    dt: float
    device: Any
    gravity: float = DEFAULT_GRAVITY
    warmup_steps: int = 50
    cmd_tol: float = 0.15
    min_cmd_speed: float = 0.1
    leg_length_m: Optional[float] = None  # for Froude number; optional
    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        N, dev = self.num_envs, self.device
        z = lambda: torch.zeros(N, device=dev, dtype=torch.float64)
        # energy / distance integrals (per env)
        self._energy = z()          # Σ |τ·q̇| dt
        self._distance = z()        # Σ ||v_xy|| dt
        self._speed_sum = z()       # Σ v_x   (forward speed, body frame)
        self._active_steps = z()    # counted steps per env (alive)
        # tracking (post-warmup, alive)
        self._lin_err_sq = z()      # Σ ||v_xy - cmd_xy||²
        self._ang_err_sq = z()      # Σ (w_z - cmd_z)²
        self._track_success = z()   # #steps within tolerance
        self._track_steps = z()     # #steps counted for tracking
        # survival
        self._alive = torch.ones(N, device=dev, dtype=torch.bool)
        self._fall_step = torch.full((N,), -1, device=dev, dtype=torch.int64)
        self._t = 0

    # ------------------------------------------------------------------ update
    def update(
        self,
        *,
        base_lin_vel: torch.Tensor,   # (N,3) body frame
        base_ang_vel: torch.Tensor,   # (N,3) body frame
        commands: torch.Tensor,       # (N,>=3): [lin_x, lin_y, ang_z]
        control_force: torch.Tensor,  # (N,dof) joint torque
        dof_vel: torch.Tensor,        # (N,dof) joint velocity
        fell: Optional[torch.Tensor] = None,   # (N,) bool: failed (not timeout) this step
        active: Optional[torch.Tensor] = None,  # (N,) bool: count this env this step
    ) -> None:
        dev = self.device
        blv = base_lin_vel.to(dev).double()
        bav = base_ang_vel.to(dev).double()
        cmd = commands.to(dev).double()
        tau = control_force.to(dev).double()
        qd = dof_vel.to(dev).double()

        alive = self._alive.clone()
        if active is not None:
            alive = alive & active.to(dev).bool()
        alive_f = alive.double()

        # --- energy & distance ---
        power = (tau * qd).abs().sum(dim=1)         # |τ·q̇|
        self._energy += power * self.dt * alive_f
        speed_xy = blv[:, :2].norm(dim=1)
        self._distance += speed_xy * self.dt * alive_f
        self._speed_sum += blv[:, 0] * alive_f
        self._active_steps += alive_f

        # --- velocity tracking (ignore warmup) ---
        if self._t >= self.warmup_steps:
            lin_err_sq = (blv[:, :2] - cmd[:, :2]).pow(2).sum(dim=1)
            ang_err_sq = (bav[:, 2] - cmd[:, 2]).pow(2)
            self._lin_err_sq += lin_err_sq * alive_f
            self._ang_err_sq += ang_err_sq * alive_f
            denom = torch.clamp(cmd[:, 0].abs(), min=self.min_cmd_speed)
            success = ((blv[:, 0] - cmd[:, 0]).abs() <= self.cmd_tol * denom).double()
            self._track_success += success * alive_f
            self._track_steps += alive_f

        # --- survival ---
        if fell is not None:
            fell_b = fell.to(dev).bool() & self._alive
            newly = fell_b.nonzero(as_tuple=False).flatten()
            if newly.numel() > 0:
                self._alive[newly] = False
                self._fall_step[newly] = int(self._t)

        self._t += 1

    def update_from_env(
        self,
        env,
        dones: Optional[torch.Tensor] = None,
        infos: Optional[Dict[str, Any]] = None,
        active: Optional[torch.Tensor] = None,
    ) -> None:
        """Convenience for ``Go2Env``-style envs (shared obs/attr layout)."""
        control_force = env.robot.get_dofs_control_force(env.motors_dof_idx)
        dof_vel = env.dof_vel
        fell = None
        if dones is not None:
            dones_b = dones.to(self.device).bool()
            time_outs = None
            if infos is not None and "time_outs" in infos:
                time_outs = infos["time_outs"].to(self.device).bool()
            elif "time_outs" in getattr(env, "extras", {}):
                time_outs = env.extras["time_outs"].to(self.device).bool()
            fell = dones_b & (~time_outs) if time_outs is not None else dones_b
        self.update(
            base_lin_vel=env.base_lin_vel,
            base_ang_vel=env.base_ang_vel,
            commands=env.commands,
            control_force=control_force,
            dof_vel=dof_vel,
            fell=fell,
            active=active,
        )

    # ----------------------------------------------------------------- compute
    def compute(self) -> Dict[str, float]:
        mg = self.robot_mass * self.gravity
        moved = self._distance > (self.min_cmd_speed * self.dt)  # envs that actually moved
        # CoT per env (only where it moved), then mean.
        cot = torch.full_like(self._distance, float("nan"))
        denom = mg * self._distance
        valid = moved & (denom > self.eps)
        cot[valid] = self._energy[valid] / denom[valid]
        cot_mean = float(cot[valid].mean().item()) if valid.any() else float("nan")

        track_steps = torch.clamp(self._track_steps, min=1.0)
        lin_rmse = float((self._lin_err_sq / track_steps).mean().sqrt().item())
        ang_rmse = float((self._ang_err_sq / track_steps).mean().sqrt().item())
        success_rate = float((self._track_success / track_steps).mean().item())

        survival_rate = float(self._alive.double().mean().item())
        ep_len = torch.where(
            self._alive,
            torch.full_like(self._fall_step, self._t),
            self._fall_step,
        ).double()
        mean_ep_len = float(ep_len.mean().item())

        steps = torch.clamp(self._active_steps, min=1.0)
        avg_speed = float((self._speed_sum / steps).mean().item())
        mean_distance = float(self._distance.mean().item())

        out = {
            "cost_of_transport": cot_mean,
            "tracking_success_rate": success_rate,
            "lin_vel_rmse": lin_rmse,
            "ang_vel_rmse": ang_rmse,
            "survival_rate": survival_rate,
            "mean_episode_length": mean_ep_len,
            "mean_distance_m": mean_distance,
            "avg_forward_speed_mps": avg_speed,
            "num_envs": int(self.num_envs),
            "total_steps": int(self._t),
            "robot_mass_kg": float(self.robot_mass),
        }
        if self.leg_length_m:
            out["froude_number"] = (avg_speed ** 2) / (self.gravity * float(self.leg_length_m))
        return out

    @staticmethod
    def format_summary(m: Dict[str, float]) -> str:
        def g(k):
            v = m.get(k)
            return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.4f}"
        lines = [
            "Locomotion metrics:",
            f"  Cost of Transport (CoT):   {g('cost_of_transport')}   (lower is better)",
            f"  Tracking success rate:     {g('tracking_success_rate')}   (±{int(m.get('_tol', 15))}% cmd)"
            if "_tol" in m else f"  Tracking success rate:     {g('tracking_success_rate')}",
            f"  Lin-vel RMSE:              {g('lin_vel_rmse')} m/s",
            f"  Ang-vel RMSE:              {g('ang_vel_rmse')} rad/s",
            f"  Survival rate:             {g('survival_rate')}",
            f"  Mean episode length:       {g('mean_episode_length')} steps",
            f"  Mean distance:             {g('mean_distance_m')} m",
            f"  Avg forward speed:         {g('avg_forward_speed_mps')} m/s",
        ]
        if "froude_number" in m:
            lines.append(f"  Froude number:             {g('froude_number')}")
        return "\n".join(lines)


def robot_total_mass(env) -> float:
    """Total mass (kg) of the env's robot entity."""
    try:
        mass = env.robot.get_mass()
        return float(mass.item()) if hasattr(mass, "item") else float(mass)
    except Exception:
        # Fallback: sum inertial link masses.
        try:
            return float(sum(float(l.get_mass()) for l in env.robot.links))
        except Exception:
            return 1.0
