#!/usr/bin/env python3
"""Demo clips for Notion: Vintix / transferred expert on scaled Go1 bodies.

Option A: tilt / timeout resets are disabled for recording only, so a failing
body stays fallen and the failure mode is visible. Metrics evaluation is unchanged.

Layout (one row per axis, left -> right):
  lower still-walks | 1.0x | upper comfortable | upper borderline | upper fail
Lower "can't walk" clips are deferred until issue #13 finds that cliff.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parent
WORKSPACE = EXP_DIR.parents[1]
SCRIPTS = WORKSPACE / "vintix_go2" / "scripts"
LOCO = WORKSPACE / "Genesis" / "examples" / "locomotion"
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(WORKSPACE / "vintix_go2"))
sys.path.insert(0, str(LOCO))

from body_scale import (  # noqa: E402
    GO1_EXPERT_CKPT,
    GO1_WITHOUT_CKPT,
    scale_tag,
    scaled_go1_cfgs,
)
from scaled_go1_env import ScaledGo1Env  # noqa: E402
import genesis as gs  # noqa: E402

OUT_DIR = EXP_DIR / "figures" / "demo_videos"
FPS = 30
# ~8 s Notion-friendly clips (sim dt ~0.02 s; video length = steps / FPS).
DEFAULT_STEPS = 240
NEVER_DEG = 1e9
NEVER_S = 1e5
CAM_FWD, CAM_LAT, CAM_UP = 1.35, 0.95, 0.62

# (axis, scale, role_slug, Japanese label for Notion captions)
# Lower can't-walk omitted until #13.
CLIPS: List[Tuple[str, float, str, str]] = [
    ("mass", 0.25, "lower_ok", "下限 0.25×（まだ歩ける）"),
    ("mass", 1.0, "nominal", "1.0×（基準）"),
    ("mass", 1.1, "upper_ok", "上限 1.1×（余裕）"),
    ("mass", 1.25, "upper_edge", "上限 1.25×（ギリギリ）"),
    ("mass", 1.3, "upper_fail", "上限 1.3×（歩けない）"),
    ("size", 0.5, "lower_ok", "下限 0.5×（まだ歩ける）"),
    ("size", 1.0, "nominal", "1.0×（基準）"),
    ("size", 1.1, "upper_ok", "上限 1.1×（余裕）"),
    ("size", 1.25, "upper_edge", "上限 1.25×（ギリギリ）"),
    ("size", 1.35, "upper_fail", "上限 1.35×（歩けない）"),
]


def without_resets(env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    env_cfg = dict(env_cfg)
    env_cfg["termination_if_roll_greater_than"] = NEVER_DEG
    env_cfg["termination_if_pitch_greater_than"] = NEVER_DEG
    env_cfg["episode_length_s"] = NEVER_S
    return env_cfg


def _as_row(tensor) -> np.ndarray:
    t = tensor.detach().cpu().numpy()
    return t[0] if t.ndim == 2 else t


def follow_camera(env, smooth: Dict[str, Any], size_scale: float) -> None:
    pos = _as_row(env.robot.get_pos())
    quat = _as_row(env.robot.get_quat())
    w, x, y, z = [float(v) for v in quat]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    s_scale = max(float(size_scale), 0.4)
    offset = np.array(
        [c * CAM_FWD * s_scale - s * CAM_LAT * s_scale,
         s * CAM_FWD * s_scale + c * CAM_LAT * s_scale,
         CAM_UP * s_scale],
        dtype=np.float64,
    )
    target = pos + offset
    smooth["pos"] = target if smooth["pos"] is None else 0.8 * smooth["pos"] + 0.2 * target
    env.cam.set_pose(pos=smooth["pos"], lookat=pos + np.array([0.0, 0.0, 0.08 * s_scale]))


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _perturb_initial_state(env) -> None:
    from genesis.utils.geom import transform_quat_by_quat as transform_quat

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
    base_init = env.base_init_quat.reshape(1, -1).expand(1, -1)
    env.base_quat[env_idx] = transform_quat(base_init, quat_noise)
    env.robot.set_quat(env.base_quat[env_idx], zero_velocity=False, envs_idx=env_idx)

    dof_noise = (torch.rand(1, env.num_actions, device=gs.device) - 0.5) * 0.2
    env.dof_pos[env_idx] = env.default_dof_pos + dof_noise
    env.robot.set_dofs_position(
        position=env.dof_pos[env_idx],
        dofs_idx_local=env.motors_dof_idx,
        zero_velocity=True,
        envs_idx=env_idx,
    )


def _mp4_to_gif(mp4: Path, gif: Path, width: int = 480) -> None:
    """Palette GIF small enough for Notion (~few MB)."""
    palette = mp4.with_suffix(".palette.png")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
         "-vf", f"fps=12,scale={width}:-1:flags=lanczos,palettegen=max_colors=128",
         str(palette)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(palette),
         "-lavfi", f"fps=12,scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer",
         str(gif)],
        check=True,
    )
    palette.unlink(missing_ok=True)


def _stem(policy: str, axis: str, scale: float, role: str) -> str:
    return f"go1_{policy}_{axis}_{scale_tag(scale)}_{role}"


def record_vintix(axis: str, scale: float, role: str, label: str, steps: int, seed: int, force: bool) -> Path:
    from save_vintix import (
        VintixHistoryBuffer,
        _compute_averaged_stats_from_known_tasks,
        _get_task_name_for_robot,
    )
    from vintix.vintix import Vintix

    out_mp4 = OUT_DIR / f"{_stem('vintix', axis, scale, role)}.mp4"
    out_gif = out_mp4.with_suffix(".gif")
    if out_gif.exists() and not force:
        print(f"skip existing {out_gif}")
        return out_gif

    _seed_everything(seed)
    gs.init(backend=gs.gpu, logging_level="warning", seed=seed)

    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = scaled_go1_cfgs(axis, scale)
    env_cfg = without_resets(env_cfg)
    size_scale = float(env_cfg.get("body_size_scale", 1.0))

    env = ScaledGo1Env(1, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False)
    ctx = getattr(getattr(env.scene, "visualizer", None), "context", None)
    if ctx is not None and hasattr(ctx, "off_world_frame"):
        ctx.off_world_frame()

    model = Vintix()
    model.load_model(str(GO1_WITHOUT_CKPT))
    model = model.to(gs.device)
    model.eval()
    task_name, is_unknown = _get_task_name_for_robot("go1", model.metadata)
    if is_unknown:
        stats = _compute_averaged_stats_from_known_tasks(model.metadata, task_name)
        model.add_task(task_name, "quadruped_locomotion", stats, rew_scale=1.0)

    history = VintixHistoryBuffer(max_len=2048, task_name=task_name, body_param=None)
    obs, _ = env.reset()
    obs = obs[:, :-12]
    _perturb_initial_state(env)
    zero = torch.zeros(1, env.num_actions, device=gs.device)
    obs, _, _, _ = env.step(zero)
    obs = obs[:, :-12]
    history.add(obs[0].cpu().numpy(), np.zeros(env.num_actions), 0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smooth = {"pos": None}
    env.cam.start_recording()
    with torch.no_grad():
        for step in range(steps):
            follow_camera(env, smooth, size_scale)
            context = history.get_context(2048)
            if context is None:
                action = torch.zeros(1, env.num_actions, device=gs.device)
            else:
                for key in context[0]:
                    if isinstance(context[0][key], torch.Tensor):
                        context[0][key] = context[0][key].to(gs.device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred, _ = model(context)
                if isinstance(pred, list):
                    pred = pred[0]
                if pred.dim() == 3:
                    action = pred[0, -1, :].unsqueeze(0).float()
                elif pred.dim() == 2:
                    action = pred[-1, :].unsqueeze(0).float()
                else:
                    raise ValueError(pred.shape)
            obs, rewards, _, _ = env.step(action)
            obs = obs[:, :-12]
            env.cam.render()
            history.add(obs[0].cpu().numpy(), action[0].cpu().numpy(), float(rewards.cpu().numpy()[0]))
            if (step + 1) % 60 == 0:
                print(f"  vintix {axis} x{scale:g} step {step + 1}/{steps}")

    env.cam.stop_recording(save_to_filename=str(out_mp4), fps=FPS)
    _mp4_to_gif(out_mp4, out_gif)
    meta = {
        "policy": "vintix",
        "axis": axis,
        "scale": scale,
        "role": role,
        "label": label,
        "steps": steps,
        "fps": FPS,
        "seed": seed,
        "resets": "disabled (demo only)",
        "model": str(GO1_WITHOUT_CKPT),
        "mp4": out_mp4.name,
        "gif": out_gif.name,
        "gif_mb": round(out_gif.stat().st_size / 1e6, 2),
    }
    out_mp4.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"saved {out_gif} ({meta['gif_mb']} MB)")
    return out_gif


def record_expert(axis: str, scale: float, role: str, label: str, steps: int, seed: int, force: bool) -> Path:
    import copy

    from rsl_rl.runners import OnPolicyRunner

    out_mp4 = OUT_DIR / f"{_stem('expert', axis, scale, role)}.mp4"
    out_gif = out_mp4.with_suffix(".gif")
    if out_gif.exists() and not force:
        print(f"skip existing {out_gif}")
        return out_gif

    _seed_everything(seed)
    gs.init(backend=gs.gpu, logging_level="warning", seed=seed)

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = scaled_go1_cfgs(axis, scale)
    env_cfg = without_resets(env_cfg)
    size_scale = float(env_cfg.get("body_size_scale", 1.0))
    env = ScaledGo1Env(1, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False)
    ctx = getattr(getattr(env.scene, "visualizer", None), "context", None)
    if ctx is not None and hasattr(ctx, "off_world_frame"):
        ctx.off_world_frame()

    train_cfg = copy.deepcopy(train_cfg)
    log_dir = str(WORKSPACE / "Genesis" / "logs" / "go1-walking")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(str(GO1_EXPERT_CKPT))
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    _perturb_initial_state(env)
    zero = torch.zeros(1, env.num_actions, device=gs.device)
    obs, _, _, _ = env.step(zero)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smooth = {"pos": None}
    env.cam.start_recording()
    with torch.no_grad():
        for step in range(steps):
            follow_camera(env, smooth, size_scale)
            action = policy(obs)
            obs, _, _, _ = env.step(action)
            env.cam.render()
            if (step + 1) % 60 == 0:
                print(f"  expert {axis} x{scale:g} step {step + 1}/{steps}")

    env.cam.stop_recording(save_to_filename=str(out_mp4), fps=FPS)
    _mp4_to_gif(out_mp4, out_gif)
    meta = {
        "policy": "expert",
        "axis": axis,
        "scale": scale,
        "role": role,
        "label": label,
        "steps": steps,
        "fps": FPS,
        "seed": seed,
        "resets": "disabled (demo only)",
        "model": str(GO1_EXPERT_CKPT),
        "mp4": out_mp4.name,
        "gif": out_gif.name,
        "gif_mb": round(out_gif.stat().st_size / 1e6, 2),
    }
    out_mp4.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"saved {out_gif} ({meta['gif_mb']} MB)")
    return out_gif


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=("vintix", "expert", "all"), default="vintix")
    p.add_argument("--axis", choices=("mass", "size", "all"), default="all")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--one", type=str, default=None,
                   help="Record a single clip: axis:scale:role e.g. mass:1.25:upper_edge")
    args = p.parse_args()

    clips = CLIPS
    if args.one:
        axis, scale_s, role = args.one.split(":")
        scale = float(scale_s)
        label = next((lab for a, s, r, lab in CLIPS if a == axis and abs(s - scale) < 1e-9 and r == role), args.one)
        clips = [(axis, scale, role, label)]
    else:
        if args.axis != "all":
            clips = [c for c in clips if c[0] == args.axis]

    policies = ("vintix", "expert") if args.policy == "all" else (args.policy,)
    # One process = one Genesis init. Call this script once per clip from a driver, or loop
    # here knowing only the first gs.init in the process succeeds reliably.
    if len(clips) * len(policies) > 1 and args.one is None:
        # Re-exec each clip in a fresh interpreter (Genesis init is once-per-process).
        for policy in policies:
            for axis, scale, role, _ in clips:
                cmd = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--policy", policy, "--one", f"{axis}:{scale:g}:{role}",
                    "--steps", str(args.steps), "--seed", str(args.seed),
                ]
                if args.force:
                    cmd.append("--force")
                print("RUN", " ".join(cmd))
                subprocess.run(cmd, check=True, cwd=str(WORKSPACE / "vintix_go2"))
        return

    axis, scale, role, label = clips[0]
    policy = policies[0]
    print(f"Recording {policy} {axis} x{scale:g} ({role}) steps={args.steps} seed={args.seed}")
    if policy == "vintix":
        record_vintix(axis, scale, role, label, args.steps, args.seed, args.force)
    else:
        record_expert(axis, scale, role, label, args.steps, args.seed, args.force)


if __name__ == "__main__":
    main()
