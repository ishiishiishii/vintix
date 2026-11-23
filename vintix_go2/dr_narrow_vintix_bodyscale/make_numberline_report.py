#!/usr/bin/env python3
"""Number-line figures (train morphology range vs Go1 eval cliff) + REPORT.md (#21).

Physical units on the number line (kg, m) — unlike scale-axis summary plots which
use multipliers only (figure-style.md). Eval cliffs use Go1 nominal mass/thigh.
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
VINTIX_ROOT = EXP_DIR.parent
RB_DIR = VINTIX_ROOT / "robot_body_scale_zeroshot"
GO1_DIR = VINTIX_ROOT / "go1_body_scale_zeroshot"
META_PATH = EXP_DIR / "configs" / "dr_narrow_success_meta.json"
FIG_DIR = EXP_DIR / "figures"
REPORT_PATH = EXP_DIR / "REPORT.md"
NOTION_PATH = EXP_DIR / "NOTION.md"
SUCCESS_THRESH = 50.0  # Success rate [%] cliff threshold


def go1_nominal_mass() -> float:
    urdf = Path("/workspace/Genesis/genesis/assets/urdf/go1/urdf/go1.urdf")
    root = ET.parse(urdf).getroot()
    return sum(float(m.get("value", 0)) for m in root.iter("mass"))


def go1_thigh_m() -> float:
    import sys
    sys.path.insert(0, str(RB_DIR))
    from robots import ROBOTS  # noqa: WPS433
    return float(ROBOTS["go1"]["thigh_m"])


def load_eval_completion() -> Dict[str, Dict[float, List[float]]]:
    """axis -> scale -> list of per-seed completion %."""
    import sys
    sys.path.insert(0, str(GO1_DIR))
    sys.path.insert(0, str(RB_DIR))
    import run_experiment as rx  # noqa: E402
    from body_scale import FULL_GRID as GO1_FULL_GRID  # noqa: E402
    from robots import EVAL_SEEDS, GRIDS, canonical_axis, scale_tag  # noqa: E402

    rx.RESULTS_DIR = EXP_DIR / "results"
    grids = {
        "mass": GO1_FULL_GRID["mass"],
        "size": GO1_FULL_GRID["size"],
        "size_density": GRIDS["size_density"],
    }
    out: Dict[str, Dict[float, List[float]]] = {a: {} for a in grids}
    for axis, scales in grids.items():
        for scale in scales:
            if abs(scale - 1.0) < 1e-9 and axis != "mass":
                continue
            vals = []
            for seed in EVAL_SEEDS:
                csv_path = rx.episodes_csv("go1", "vintix", axis, scale, seed)
                if not csv_path.exists():
                    continue
                with csv_path.open() as f:
                    rows = list(csv.DictReader(f))
                if not rows:
                    continue
                # prefer explicit completion column if present
                if "completed" in rows[0] or "completion" in rows[0]:
                    key = "completed" if "completed" in rows[0] else "completion"
                    c = np.mean([float(r[key]) for r in rows])
                    if c <= 1.0:
                        c *= 100.0
                else:
                    lengths = [float(r.get("episode_length", r.get("length", 0))) for r in rows]
                    c = float(np.mean([1.0 if L >= 1000 else 0.0 for L in lengths]) * 100.0)
                vals.append(c)
            if vals:
                out[axis][float(scale)] = vals
    return out


def cliff_range(scale_to_vals: Dict[float, List[float]], thresh: float = SUCCESS_THRESH
                ) -> Tuple[Optional[float], Optional[float]]:
    """Return (max walkable scale below/near 1.0 side? , first failing).

    Walkable := mean completion >= thresh. Returns (last_ok, first_fail) over
    sorted scales (ascending).
    """
    if not scale_to_vals:
        return None, None
    scales = sorted(scale_to_vals)
    means = {s: float(np.mean(scale_to_vals[s])) for s in scales}
    last_ok = None
    first_fail = None
    for s in scales:
        if means[s] >= thresh:
            last_ok = s
        elif first_fail is None and last_ok is not None:
            first_fail = s
            break
    # also capture failure before any success (all fail)
    if last_ok is None:
        first_fail = scales[0]
    return last_ok, first_fail


def plot_numberline(
    values: List[float],
    *,
    xlabel: str,
    out: Path,
    eval_points: Optional[List[Tuple[float, float]]] = None,
    train_label: str = "Train robots (success)",
    fail_x: Optional[float] = None,
    baseline_x: Optional[float] = None,
    baseline_label: str = "Go1 base (1.0×)",
) -> Path:
    """Horizontal number line: train range, Go1 base, optional eval markers."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3.0), dpi=150)
    y0 = 0.0
    xs_all = list(values)
    if eval_points:
        xs_all.extend(p[0] for p in eval_points)
    if baseline_x is not None:
        xs_all.append(baseline_x)
    if fail_x is not None:
        xs_all.append(fail_x)
    x_lo = min(xs_all) * 0.95 if xs_all else 0.0
    x_hi = max(xs_all) * 1.05 if xs_all else 1.0
    ax.hlines(y0, x_lo, x_hi, colors="#444444", linewidths=2.0, zorder=1)
    if values:
        ax.scatter(values, [y0] * len(values), s=28, c="#1f77b4", alpha=0.55,
                   label=train_label, zorder=3, edgecolors="none")
        ax.plot([min(values), max(values)], [y0, y0], color="#1f77b4", linewidth=6,
                alpha=0.25, solid_capstyle="butt", zorder=2)
    if eval_points:
        xs = [p[0] for p in eval_points]
        cs = [p[1] for p in eval_points]
        colors = ["#2ca02c" if c >= SUCCESS_THRESH else "#d62728" for c in cs]
        ax.scatter(xs, [0.14] * len(xs), s=42, c=colors, marker="D",
                   label=f"Go1 eval (green ≥ {SUCCESS_THRESH:.0f}% success)", zorder=4,
                   edgecolors="#222222", linewidths=0.4)
    if baseline_x is not None:
        ax.axvline(baseline_x, color="#111111", linestyle="-", linewidth=2.0, zorder=6)
        ax.scatter([baseline_x], [-0.16], s=90, c="#111111", marker="o",
                   label=f"{baseline_label}", zorder=7,
                   edgecolors="#ffffff", linewidths=1.0)
        # value callout just above the axis (readable, not a title)
        ax.annotate(
            f"{baseline_x:.3g}",
            xy=(baseline_x, y0),
            xytext=(0, -28),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=12,
            color="#111111",
        )
    if fail_x is not None:
        ax.axvline(fail_x, color="#d62728", linestyle="--", linewidth=1.5,
                   label=f"Fails near {fail_x:.3g}", zorder=5)
    ax.set_ylim(-0.42, 0.48)
    ax.set_xlim(x_lo, x_hi)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=11, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.48)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def write_report(meta: dict, eval_comp: Dict[str, Dict[float, List[float]]], figs: List[Path]) -> None:
    robots = meta.get("robots", [])
    masses = [r["total_mass_kg"] for r in robots]
    legs = [r["leg_length_m"] for r in robots]
    notion_url = ""
    if NOTION_PATH.exists():
        notion_url = NOTION_PATH.read_text().strip().splitlines()[0].strip()

    n_eval = sum(len(v) for v in eval_comp.values())
    status = "評価完了" if n_eval else "学習待ち／評価前"

    lines = [
        "# DR narrow 履歴 Vintix の体スケール限界 — 実験レポート",
        "",
        "> やったこと: narrow-DR 成功体の PPO 履歴で Vintix を学習し、Go1 の質量／大きさ／密度一定の 3 軸で評価する。",
        "> これから: 数直線上で学習分布と崖を対応付け、同体学習（#18）・leave-one-out（#10）と比較する。",
        "",
        "---",
        "",
        "### 概要",
        f"narrow 100 体のうち報酬 ≥15 の **{meta.get('n_success', len(robots))} 体**の履歴で Vintix を学習し、",
        "Go1 体スケール 3 軸でゼロショット評価する。学習に使った質量・脚長の範囲を数直線に示し、",
        "どこまで歩けてどこから無理かを対応付ける。",
        "",
        "### 目的",
        "形態 DR 由来の多様な履歴だけで学習した Vintix が、Go1 の質量・大きさ変化にどこまで耐えるか。",
        "学習分布の外側／内側で崖がどう出るかを数直線で見る。",
        "",
        "### 方法",
        "- データ: `dr_robots_ppo_history` narrow GenLoco、成功体のみ（reward_last ≥ 15）",
        "- 学習: `scripts/train_vintix.py`（epochs=5, context=2048, sparsity=128, group=`quadruped_locomotion`）",
        "- 評価: #18 と同じ Go1 3 軸格子・10 seeds（Vintix のみ）",
        "- 可視化: 学習ロボットの質量・脚長数直線 ＋ 評価 Success rate の対応",
        "",
        "### 実験内容・設定",
        "",
        "| 項目 | 値 |",
        "|------|-----|",
        f"| 成功体数 | {meta.get('n_success')} / {meta.get('n_total')} |",
        f"| 質量範囲 [kg] | {min(masses):.2f} – {max(masses):.2f} |" if masses else "| 質量範囲 | n/a |",
        f"| 脚長範囲 [m] | {min(legs):.3f} – {max(legs):.3f} |" if legs else "| 脚長範囲 | n/a |",
        f"| 状態 | {status} |",
        f"| Issue | [#21](https://github.com/ishiishiishii/vintix/issues/21) |",
        "",
        "### 結果",
        "",
    ]
    if figs:
        for fig in figs:
            rel = fig.relative_to(EXP_DIR)
            lines.append(f"![{fig.stem}]({rel})")
            lines.append("")
    else:
        lines.append("（図は評価完了後に生成）")
        lines.append("")

    if n_eval:
        go1_m = go1_nominal_mass()
        lines.append("| 軸 | 最後に歩けた倍率 | 最初に落ちた倍率 | 対応する物理量（目安） |")
        lines.append("|----|------------------|------------------|------------------------|")
        for axis in ("mass", "size", "size_density"):
            last_ok, first_fail = cliff_range(eval_comp.get(axis, {}))
            if axis == "mass":
                phys = (f"mass≈{go1_m * last_ok:.2f} kg" if last_ok else "—")
            else:
                th = go1_thigh_m()
                phys = (f"thigh≈{th * last_ok:.3f} m" if last_ok else "—")
            lines.append(
                f"| {axis} | {last_ok if last_ok is not None else '—'} | "
                f"{first_fail if first_fail is not None else '—'} | {phys} |"
            )
        lines.append("")
    else:
        lines.append("評価結果はまだ無い（Go1 同体評価 #18 完了後に学習→評価）。")
        lines.append("")

    lines += [
        "### うまくいかなかったこと → 修正",
        "- （実行後に追記）",
        "",
        "### 考察・いま言えること",
        "- 学習分布（数直線の青帯）の内側でも Go1 評価が落ちるなら、分布カバレッジだけでは足りない。",
        "- 分布の外で落ちるなら、まず範囲拡張（wide / 追加体）が次の一手。",
        "",
        "### 次の予定",
        "- #18 完了待ち → 本実験の学習・3 軸評価・数直線レポート更新",
        "- 同体学習・leave-one-out との崖位置比較",
        "",
        "<details>",
        "<summary>詳細</summary>",
        "",
        "#### データフィルタ",
        f"- CSV: `dr_robots_ppo_history/outputs/dr_narrow_expert_rewards.csv`",
        f"- 閾値: reward_last ≥ {meta.get('reward_min', 15)}",
        f"- メタ: `configs/dr_narrow_success_meta.json`",
        "",
        "#### 再現",
        "```bash",
        "cd /workspace/vintix_go2",
        "bash dr_narrow_vintix_bodyscale/wait_and_run.sh   # #18 完了後に学習→評価→レポート",
        "python dr_narrow_vintix_bodyscale/make_numberline_report.py",
        "```",
        "",
        "</details>",
        "",
        "---",
        f"Notion: {notion_url or '(page pending)'}",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    print(f"wrote {REPORT_PATH}")


def main() -> None:
    if not META_PATH.exists():
        # allow report scaffold before build_dataset
        meta = {"n_success": 0, "n_total": 100, "reward_min": 15.0, "robots": []}
    else:
        meta = json.loads(META_PATH.read_text())

    robots = meta.get("robots", [])
    masses = [r["total_mass_kg"] for r in robots]
    legs = [r["leg_length_m"] for r in robots]
    figs: List[Path] = []

    eval_comp: Dict[str, Dict[float, List[float]]] = {}
    try:
        eval_comp = load_eval_completion()
    except Exception as e:  # noqa: BLE001 — report still useful pre-eval
        print(f"[warn] eval load skipped: {e}")

    go1_m = go1_nominal_mass()
    go1_th = go1_thigh_m()
    # DR meta stores thigh+calf as leg_length; Go1 calf ≈ thigh in the URDF table.
    go1_leg = 2.0 * go1_th

    mass_eval = None
    fail_mass = None
    if eval_comp.get("mass"):
        mass_eval = [(go1_m * s, float(np.mean(vs))) for s, vs in sorted(eval_comp["mass"].items())]
        _ok, fail = cliff_range(eval_comp["mass"])
        fail_mass = go1_m * fail if fail is not None else None

    if masses:
        figs.append(plot_numberline(
            masses,
            xlabel="Total mass [kg]",
            out=FIG_DIR / "train_mass_numberline_vs_go1_eval.png",
            eval_points=mass_eval,
            fail_x=fail_mass,
            baseline_x=go1_m,
            baseline_label=f"Go1 base mass ({go1_m:.2f} kg)",
        ))

    size_eval = None
    fail_leg = None
    # Same physical unit as train: Go1 leg_length ≈ 2 × thigh, then × size scale.
    size_scales = {}
    for axis in ("size", "size_density"):
        for s, vs in eval_comp.get(axis, {}).items():
            size_scales.setdefault(s, []).extend(vs)
    if size_scales:
        size_eval = [(go1_leg * s, float(np.mean(vs))) for s, vs in sorted(size_scales.items())]
        _ok, fail = cliff_range(size_scales)
        fail_leg = go1_leg * fail if fail is not None else None

    if legs:
        figs.append(plot_numberline(
            legs,
            xlabel="Leg length [m] (thigh + calf)",
            out=FIG_DIR / "train_leg_numberline_vs_go1_eval.png",
            eval_points=size_eval,
            fail_x=fail_leg,
            baseline_x=go1_leg,
            baseline_label=f"Go1 base leg ({go1_leg:.3f} m)",
            train_label="Train robots (leg length)",
        ))

    write_report(meta, eval_comp, figs)
    for f in figs:
        print(f)


if __name__ == "__main__":
    main()
