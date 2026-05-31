"""Shared plot style aligned with ``ppo_leave_one_out_20260519/EXPERIMENT.md``."""

from __future__ import annotations

# ppo_leave_one_out_20260519/run_experiment.py (plot_all_robots)
Y_LIM = (-5.0, 28.0)
Y_LABEL = "Mean Cumulative Reward"
STD_FILL_ALPHA = 0.15
LINE_WIDTH = 2.6
FIGSIZE = (14, 10)
LEGEND_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 22
AXIS_TICK_FONTSIZE = 16

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


def shade_mean_std(
    ax,
    x,
    mean,
    std,
    *,
    color: str,
    y_lim: tuple[float, float] = Y_LIM,
    alpha: float = STD_FILL_ALPHA,
) -> None:
    """Mean line with ±1 std band clipped to ``y_lim`` (PPO leave-one-out style)."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    lower = np.maximum(mean - std, y_lim[0])
    upper = np.minimum(mean + std, y_lim[1])
    ax.fill_between(x, lower, upper, alpha=alpha, color=color)
