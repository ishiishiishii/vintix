"""Shared plot style aligned with ``ppo_leave_one_out_20260519/EXPERIMENT.md``."""

from __future__ import annotations

# ppo_leave_one_out_20260519/run_experiment.py
Y_LIM = (-5.0, 28.0)
Y_LABEL = "Mean Cumulative Reward"

ROBOT_COLORS: dict[str, str] = {
    "go1": "#1f77b4",
    "go2": "#ff7f0e",
    "a1": "#2ca02c",
    "minicheetah": "#d62728",
}

ROBOT_LABELS: dict[str, str] = {
    "go1": "Go1",
    "go2": "Go2",
    "a1": "A1",
    "minicheetah": "MiniCheetah",
}

CURVE_COLORS_ORDER = ("go1", "go2", "a1", "minicheetah")


def finetune_legend_label(robot: str) -> str:
    return f"Finetune {ROBOT_LABELS.get(robot, robot)}"
