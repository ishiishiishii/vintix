#!/usr/bin/env python3
"""Expert quality eval for #16: completion, forward speed, tracking RMSE, optional video.

Does NOT judge by return alone (reward scales differ across robots).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Genesis" / "examples" / "locomotion"))

import genesis as gs  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def _load_cfgs(robot: str):
    from train import (
        get_aliengo_cfgs,
        get_anymalb_cfgs,
        get_b1_cfgs,
        get_go1_cfgs,
        get_go2_cfgs,
        get_laikago_cfgs,
        get_spot_cfgs,
        get_train_cfg,
    )
    from env import (
        AliengoEnv,
        ANYmalBEnv,
        B1Env,
        Go1Env,
        Go2Env,
        LaikagoEnv,
        SpotEnv,
    )

    getters = {
        "go2": (get_go2_cfgs, Go2Env),
        "go1": (get_go1_cfgs, Go1Env),
        "aliengo": (get_aliengo_cfgs, AliengoEnv),
        "b1": (get_b1_cfgs, B1Env),
        "anymalb": (get_anymalb_cfgs, ANYmalBEnv),
        "spot": (get_spot_cfgs, SpotEnv),
        "laikago": (get_laikago_cfgs, LaikagoEnv),
    }
    if robot not in getters:
        raise SystemExit(f"unsupported robot: {robot}")
    get_cfgs, EnvCls = getters[robot]
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    return env_cfg, obs_cfg, reward_cfg, command_cfg, EnvCls, get_train_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--robot", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--num_envs", type=int, default=10)
    ap.add_argument("--rollout_steps", type=int, default=1000)
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--video", type=str, default="", help="optional mp4 path (1 env camera)")
    ap.add_argument("--cmd_vx", type=float, default=None, help="override lin_vel_x command")
    args = ap.parse_args()

    env_cfg, obs_cfg, reward_cfg, command_cfg, EnvCls, get_train_cfg = _load_cfgs(args.robot)
    log_dir = str(Path(args.model_path).resolve().parent)
    # Prefer training-time cfgs so PD/height retunes match the checkpoint physics.
    cfgs_pkl = Path(log_dir) / "cfgs.pkl"
    if cfgs_pkl.is_file():
        loaded = pickle.load(open(cfgs_pkl, "rb"))
        env_cfg, obs_cfg, reward_cfg, command_cfg = loaded[0], loaded[1], loaded[2], loaded[3]
    if args.cmd_vx is not None:
        command_cfg = dict(command_cfg)
        command_cfg["lin_vel_x_range"] = [args.cmd_vx, args.cmd_vx]

    cmd_vx = float(command_cfg["lin_vel_x_range"][0])
    gs.init(logging_level="warning", precision="64")

    n = args.num_envs
    env = EnvCls(
        num_envs=n,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )
    train_cfg = get_train_cfg("expert-quality", args.rollout_steps, seed=1)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(args.model_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    record = bool(args.video)
    if record:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        env.cam.start_recording()

    ep_len = torch.zeros(n, device=gs.device)
    ep_ret = torch.zeros(n, device=gs.device)
    finished_lens, finished_rets = [], []
    vx_err_sq = []
    vx_meas = []
    alive = torch.ones(n, dtype=torch.bool, device=gs.device)

    for t in range(args.rollout_steps):
        with torch.no_grad():
            act = policy(obs)
        obs, rew, reset, extras = env.step(act)
        if isinstance(obs, tuple):
            obs = obs[0]
        # base lin vel in world or body — prefer env.base_lin_vel if present
        if hasattr(env, "base_lin_vel"):
            vx = env.base_lin_vel[:, 0]
        else:
            vx = env.base_pos.new_zeros(n)  # fallback
        vx_meas.append(vx.detach().float().cpu().numpy())
        vx_err_sq.append(((vx - cmd_vx) ** 2).detach().float().cpu().numpy())

        ep_len = torch.where(alive, ep_len + 1, ep_len)
        ep_ret = torch.where(alive, ep_ret + rew, ep_ret)
        done = reset & alive
        if done.any():
            finished_lens += ep_len[done].tolist()
            finished_rets += ep_ret[done].tolist()
            ep_len = torch.where(done, torch.zeros_like(ep_len), ep_len)
            ep_ret = torch.where(done, torch.zeros_like(ep_ret), ep_ret)
        alive = alive & (~reset)

        if record and t % 1 == 0:
            env.cam.render()

    if alive.any():
        finished_lens += ep_len[alive].tolist()
        finished_rets += ep_ret[alive].tolist()

    if record:
        env.cam.stop_recording(save_to_filename=str(args.video), fps=int(1.0 / env.dt))

    vx_all = np.concatenate(vx_meas)
    err_all = np.concatenate(vx_err_sq)
    mean_len = float(np.mean(finished_lens))
    completion = float(np.mean([L >= args.rollout_steps - 1 for L in finished_lens]))
    mean_vx = float(np.mean(vx_all))
    rmse_vx = float(np.sqrt(np.mean(err_all)))
    speed_ratio = mean_vx / cmd_vx if abs(cmd_vx) > 1e-6 else float("nan")

    payload = {
        "robot": args.robot,
        "model_path": str(args.model_path),
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_envs": n,
        "rollout_steps": args.rollout_steps,
        "cmd_vx": cmd_vx,
        "completion_rate": completion,
        "mean_episode_length": mean_len,
        "mean_return": float(np.mean(finished_rets)),
        "mean_forward_vx": mean_vx,
        "vx_tracking_rmse": rmse_vx,
        "speed_ratio_vs_cmd": speed_ratio,
        "pass_completion": completion >= 0.95,
        "pass_speed": speed_ratio >= 0.95 if abs(cmd_vx) > 1e-6 else False,
        "video": args.video or None,
        "note": "Judge by completion + speed tracking; return is reference only.",
    }
    text = json.dumps(payload, indent=2) + "\n"
    print(text)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)


if __name__ == "__main__":
    main()
